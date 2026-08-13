#!/usr/bin/env python3
"""Offline PPO retraining for the unified multi-task MoE agent.

Reads per-task training buffers (`training_<short>_<wid>.npy`) produced by
`criticality/test/test_model.py --training_out ...` and fine-tunes a
`MultiTaskAgent` with PPO-clip + BC anchor + value loss, using importance
weights from the NADE sampler.

Buffer files must be named `training_<short>_<wid>.npy` where `<short>` is
one of `stack / peg` (task_registry.short_name). Files not
matching the pattern are skipped with a warning.

Per-task caches are saved as `all_data_unified_weight_<short>.npy` inside
each `--dataset` dir; delete them to force a rebuild.
"""
import os, sys
import glob
import argparse
import re
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# make `examples.baselines.ppo.*` importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.baselines.ppo.multitask_agent import (
    MultiTaskAgent, init_experts_from_per_task_ckpts,
)
from examples.baselines.ppo.task_registry import TASKS, by_short_name


# ==========================================
# 1. Data processing & loading (routed by task)
# ==========================================

_FILE_RE = re.compile(r"^training_(?P<short>[a-zA-Z]+)_\d+\.npy$")


def _infer_task_from_filename(filename: str):
    """Return TaskSpec or None (unrecognized files are silently skipped)."""
    m = _FILE_RE.match(filename)
    if m is None:
        return None
    short = m.group("short")
    try:
        return by_short_name(short)
    except KeyError:
        return None


def _empty_arrays(obs_dim: int, act_dim: int):
    return dict(
        obs=np.empty((0, obs_dim), dtype=np.float32),
        actions=np.empty((0, act_dim), dtype=np.float32),
        returns=np.empty((0,), dtype=np.float32),
        weights=np.empty((0,), dtype=np.float32),
        log_prob=np.empty((0,), dtype=np.float32),
    )


def _build_task_cache(data_dir: str, short: str, files: list, gamma: float,
                      obs_dim: int, act_dim: int) -> dict:
    """Build cached data for one task from a list of raw .npy files."""
    tmp = _empty_arrays(obs_dim, act_dim)
    for filename in files:
        path = os.path.join(data_dir, filename)
        try:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.ndarray):
                data = data.item()
        except Exception as e:
            print(f"  [skip] {filename}: load failed ({e})")
            continue
        obs = np.array(data['obs'], dtype=np.float32)
        if obs.shape[1] < 48:
            pad_width = 48 - obs.shape[1]
            obs = np.pad(obs, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
        acts = np.array(data['actions'], dtype=np.float32)
        rews = np.array(data['rewards'], dtype=np.float32)
        dones = np.array(data['dones'], dtype=bool)
        weights = np.array(data['weights'], dtype=np.float32)
        log_probs = np.array(data['log_probs'], dtype=np.float32).reshape(-1)

        if obs.shape[1] != obs_dim:
            print(f"  [skip] {filename}: obs_dim={obs.shape[1]} != expected {obs_dim} for task {short}")
            continue
        if acts.shape[1] != act_dim:
            print(f"  [skip] {filename}: act_dim={acts.shape[1]} != expected {act_dim}")
            continue

        # Offline computation of Discounted Returns
        returns = np.zeros_like(rews, dtype=np.float32)
        G = 0.0
        for i in reversed(range(len(rews))):
            if dones[i]:
                G = rews[i]
            else:
                G = rews[i] + gamma * G
            returns[i] = G

        tmp['obs'] = np.concatenate([tmp['obs'], obs])
        tmp['actions'] = np.concatenate([tmp['actions'], acts])
        tmp['returns'] = np.concatenate([tmp['returns'], returns])
        tmp['weights'] = np.concatenate([tmp['weights'], weights])
        tmp['log_prob'] = np.concatenate([tmp['log_prob'], log_probs])
    return tmp


def load_offline_dataset(data_dirs, gamma: float = 0.95):
    """Load all datasets by task, returning {task_id: (obs_t, acts_t, returns_t, weights_t, log_probs_t)}.

    data_dirs: str or list[str], can mix output directories from multiple rounds.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    obs_dim = 48
    act_dim = 8

    # Accumulate by task_id
    per_task = {spec.task_id: _empty_arrays(obs_dim, act_dim) for spec in TASKS}

    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"[warn] dataset dir does not exist: {data_dir}")
            continue
        # Bucket files by task
        files_by_short = defaultdict(list)
        for filename in sorted(os.listdir(data_dir)):
            if not filename.endswith('.npy'):
                continue
            if filename.startswith('all_data_unified_weight'):
                continue  # this is a cache file
            spec = _infer_task_from_filename(filename)
            if spec is None:
                print(f"  [skip] {filename}: cannot infer task from filename")
                continue
            files_by_short[spec.short_name].append(filename)

        # One cache per task
        for spec in TASKS:
            short = spec.short_name
            cache_path = os.path.join(data_dir, f"all_data_unified_weight_{short}.npy")
            if os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True).item()
                print(f"[*] {data_dir}/{short}: reading cache ({data['obs'].shape[0]} steps)")
            else:
                files = files_by_short.get(short, [])
                if not files:
                    continue
                print(f"[*] {data_dir}/{short}: building cache from {len(files)} files")
                data = _build_task_cache(data_dir, short, files, gamma,
                                         obs_dim, act_dim)
                if data['obs'].shape[0] == 0:
                    print(f"  [warn] {short}: 0 valid steps, skipping cache write")
                    continue
                np.save(cache_path, data)

            agg = per_task[spec.task_id]
            for k in ('obs', 'actions', 'returns', 'weights', 'log_prob'):
                agg[k] = np.concatenate([agg[k], data[k]])

    # Convert to tensors, normalize weights per task
    out: dict = {}
    total_steps = 0
    for tid, agg in per_task.items():
        if agg['obs'].shape[0] == 0:
            continue
        obs_t = torch.tensor(agg['obs'], dtype=torch.float32)
        acts_t = torch.tensor(agg['actions'], dtype=torch.float32)
        returns_t = torch.tensor(agg['returns'], dtype=torch.float32)
        weights_t = torch.tensor(agg['weights'], dtype=torch.float32)
        weights_t = weights_t / (weights_t.mean() + 1e-8)
        log_probs_t = torch.tensor(agg['log_prob'], dtype=torch.float32)
        out[tid] = (obs_t, acts_t, returns_t, weights_t, log_probs_t)
        short = TASKS[tid].short_name
        print(f"[*] task {tid} ({short}): {obs_t.shape[0]} steps")
        total_steps += obs_t.shape[0]
    print(f"[*] Dataset loading complete. Total steps: {total_steps}, tasks: {len(out)}")
    return out


# ==========================================
# 2. Offline training main loop (multi-task version)
# ==========================================

def _make_loaders(per_task: dict, batch_size: int) -> dict:
    """{task_id: DataLoader}, sampled by weight."""
    loaders = {}
    for tid, (obs, acts, rets, wts, lps) in per_task.items():
        ds = TensorDataset(obs, acts, rets, wts, lps)
        sample_weights = torch.clamp(wts.clone(), min=1e-2, max=5.0)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        loaders[tid] = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return loaders

def set_gate_requires_grad(agent, req: bool):
    for p in agent.actor_moe.gate.parameters():
        p.requires_grad = req
    for p in agent.critic_moe.gate.parameters():
        p.requires_grad = req
    for p in agent.gate.parameters():
        p.requires_grad = req

def train_offline(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    per_task = load_offline_dataset(args.dataset, gamma=args.gamma)
    if not per_task:
        print("[ERROR] No task has valid data, exiting.")
        return

    # parse per-task loss weights: only accept comma-separated floats by task_id order
    task_loss_weights = args.task_loss_weights if args.task_loss_weights else [1.0, 1.0]

    # normalize weights to mean=1.0 to keep loss scale stable
    vals_list = task_loss_weights
    mean_val = float(np.mean(vals_list))
    for tid in range(len(vals_list)):
        task_loss_weights[tid] = float(task_loss_weights[tid]) / mean_val

    agent = MultiTaskAgent(input_dim=48, action_dim=8).to(device)

    if args.initial_ckpt and os.path.exists(args.initial_ckpt):
        sd = torch.load(args.initial_ckpt, map_location=device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        agent.load_state_dict(sd)
        print(f"[*] Successfully loaded full MultiTaskAgent ckpt: {args.initial_ckpt}")

    if args.log_std is not None:
        with torch.no_grad():
            agent.actor_logstd.fill_(float(args.log_std))
    
    if args.freeze_gate:
        set_gate_requires_grad(agent, False)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=args.learning_rate, eps=1e-5,
    )

    loaders = _make_loaders(per_task, args.batch_size)
    present_tids = sorted(loaders.keys())
    print(f"[*] Training task_ids: {present_tids}")
    print("[*] Starting multi-task offline PPO training with Policy IS...")

    agent.train()
    for epoch in range(1, args.epochs + 1):
        epoch_v_loss = defaultdict(float)
        epoch_p_loss = defaultdict(float)
        epoch_steps = defaultdict(int)

        # Within each epoch, poll and interleave minibatch updates across all tasks
        iters = {tid: iter(ld) for tid, ld in loaders.items()}
        max_len = max(len(ld) for ld in loaders.values())
        for step_idx in range(max_len):
            # tid_order = list(iters.keys())
            # np.random.shuffle(tid_order)
            # for tid in tid_order:
            loss = None
            for tid in list(iters.keys()):
                try:
                    b_obs, b_act, b_ret, b_weights, b_log_prob = next(iters[tid])
                except StopIteration:
                    iters[tid] = iter(loaders[tid])
                    b_obs, b_act, b_ret, b_weights, b_log_prob = next(iters[tid])
                b_obs, b_act, b_ret, b_log_prob, b_weights = [
                    x.to(device) for x in [b_obs, b_act, b_ret, b_log_prob, b_weights]
                ]

                _, logp, _, values = agent.get_action_and_value(b_obs, action=b_act)
                values = values.squeeze(-1)
                v_loss = F.mse_loss(values, b_ret)

                with torch.no_grad():
                    adv = b_ret - values.detach()
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                log_ratio = logp - b_log_prob
                log_ratio = torch.clamp(log_ratio, min=-5.0, max=2.0)
                ratio = torch.exp(log_ratio)
                clip_coef = 0.2
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
                ppo_loss = -(torch.min(surr1, surr2)).mean()

                # BC anchor: fit deterministic action_mean to dataset actions
                mean_a = agent.get_action(b_obs, deterministic=True)
                anchor_loss = args.bc_coef * F.mse_loss(mean_a, b_act)

                p_loss = ppo_loss + anchor_loss
                if epoch <= args.warmup_epochs:
                    tmp_loss = args.vf_coef * v_loss * 10
                else:
                    tmp_loss = p_loss + args.vf_coef * v_loss

                epoch_v_loss[tid] += v_loss.item()
                epoch_p_loss[tid] += p_loss.item()
                epoch_steps[tid] += 1

                # Apply per-task loss weighting (e.g., by accident rate)
                tid_weight = task_loss_weights[tid]
                if loss is None:
                    loss = tmp_loss * float(tid_weight)
                else:
                    loss += tmp_loss * float(tid_weight)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        current_std = agent.actor_logstd.detach().mean().item()
        msg = f"Epoch: {epoch}/{args.epochs} | Mean Std: {current_std:.4f}"
        for tid in present_tids:
            n = max(epoch_steps[tid], 1)
            short = TASKS[tid].short_name
            msg += (f" | {short}: V={epoch_v_loss[tid] / n:.4f}"
                    f" P={epoch_p_loss[tid] / n:.4f}")
        print(msg)

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_path = os.path.join(args.out_dir, f"offline_model_ep{epoch}.pt")
            torch.save(agent.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs='+',
                        default=["/mnt/mnt1/linxuan/multitask_maniskill_data/data/training/random_round2"],
                        help="One or more directories, each containing training_<short>_<wid>.npy files")
    parser.add_argument("--initial_ckpt", type=str, default="examples/baselines/ppo/runs/multitask__ppo_multitask__1__1780644413/multitask_final_ckpt.pt", help="Full MultiTaskAgent state_dict")
    # parser.add_argument("--initial_ckpt", type=str, default="training/models/round7/offline_model_best.pt")
    parser.add_argument("--out_dir", type=str, default="./training/models/random_round2")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--vf_coef", type=float, default=1.0)
    parser.add_argument("--bc_coef", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_freq", type=int, default=3)
    parser.add_argument("--freeze_gate", default=False)
    parser.add_argument("--task_loss_weights", type=float, nargs='+', default=[1.0, 1.0],
                        help="Comma-separated list of floats by task_id order, e.g. [1.0, 1.0]. "
                             "Will be normalized to mean=1.0.")
    parser.add_argument("--log_std", default=None)

    args = parser.parse_args()
    print(args)
    train_offline(args)
