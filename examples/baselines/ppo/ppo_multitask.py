"""Unified multi-task PPO for the four ManiSkill tabletop tasks.

4 parallel ManiSkillVectorEnv instances (PushCube / PickCube / StackCube /
PegInsertionSide) feed one MoE actor-critic (MultiTaskAgent). Per-task rollout
buffers; PPO update iterates epoch x task x minibatch so each task's samples
are routed through the right per-task projection.

Expert trunks can be warm-started from per-task single-task PPO ckpts via
--init_expert_ckpts (one path per task, in TASKS order: push, pick, stack,
peg). The first layer of the old ckpts is dropped (replaced by per-task
Proj_i); only the four trunk Linears are copied.

Usage:
  python ppo_multitask.py \
    --num_envs_per_task=256 --num_eval_envs_per_task=32 \
    --total_timesteps=4_000_000 --eval_freq=10 \
    --init_expert_ckpts push.pt pick.pt stack.pt peg.pt
"""

from __future__ import annotations

from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs  # noqa: F401  (registers envs)
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from multitask_agent import MultiTaskAgent, init_experts_from_per_task_ckpts
from task_registry import TASKS, TaskSpec, num_tasks

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None
    """Multi-task ckpt to load (full MultiTaskAgent state_dict)."""

    # ---- expert warm-start ----
    init_expert_ckpts: List[str] = field(default_factory=lambda: ['/home/linxuan/Embodied/push_cube/examples/baselines/ppo/runs/PushCube-v1__ppo__1__1780301489/final_ckpt.pt', '/home/linxuan/Embodied/pick_cube/examples/baselines/ppo/runs/PickCube-v1__ppo__1__1780321332/final_ckpt.pt', '/home/linxuan/Embodied/stack_cube/examples/baselines/ppo/runs/StackCube-v1__ppo__1__1780033432/final_ckpt.pt', '/home/linxuan/Embodied/insert_tube/examples/baselines/ppo/runs/PegInsertionSide-v1__ppo__1__1780488894/final_ckpt.pt']) # field(default_factory=list)
    """One path per task in TASKS order (push, pick, stack, peg). Empty list
    or empty string disables. Use 'none' or '' at a position to skip a task."""

    # ---- env / training ----
    total_timesteps: int = 400000000
    learning_rate: float = 1e-4
    num_envs_per_task: int = 512
    num_eval_envs_per_task: int = 32
    partial_reset: bool = True
    eval_partial_reset: bool = False
    num_steps: int = 100
    num_eval_steps: int = 100
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 5

    # ---- disturbance force, per task (None => off for that task) ----
    disturb_force_mag: float = 1.0
    """Global multiplier on per-task disturbance magnitude. The actual force
    applied to task i is `disturb_force_mag * TASKS[i].force_mag`. Set to 0
    to disable disturbance for all tasks."""
    disturb_prob: float = 1.0
    disturb_xy_only: bool = False
    """Force PushCube to xy-only is implicit (force_dim=2); this flag zeros z
    for the 3D tasks too if you want."""

    control_mode: Optional[str] = "pd_joint_delta_pos"
    anneal_lr: bool = False
    gamma: float = 0.95
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    reward_scale: float = 1.0
    eval_freq: int = 10
    save_train_video_freq: Optional[int] = None
    finite_horizon_gae: bool = False

    # ---- MoE auxiliary ----
    load_balance_coef: float = 0.01
    """Coefficient on Switch-Transformer-style load-balancing loss."""

    # ---- auxiliary task supervision loss ----
    task_loss_coef: float = 0.05  # cross-entropy on gate logits to predict task id

    # ---- freezing schedule ----
    freeze_expert_steps: int = 100000000  # if >0, keep expert trunks frozen for this many env steps

    # ---- runtime ----
    batch_size_per_task: int = 0
    minibatch_size_per_task: int = 0
    num_iterations: int = 0


# ----------------------------- env factory -----------------------------

def make_envs_for_task(spec: TaskSpec, num_envs: int, reconfig: Optional[int],
                       env_kwargs: dict, record_video_dir: Optional[str] = None,
                       save_train_video_trigger=None, max_steps_per_video: int = 100,
                       record_trajectory: bool = False, partial_reset: bool = True
                       ) -> Tuple[ManiSkillVectorEnv, gym.Env]:
    """Build (vector_env, base_gym_env) for one task. Returns both so caller
    can grab unwrapped attributes (apply_force target)."""
    env = gym.make(spec.env_id, num_envs=num_envs,
                   reconfiguration_freq=reconfig, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if record_video_dir is not None:
        env = RecordEpisode(env, output_dir=record_video_dir,
                            save_trajectory=record_trajectory,
                            trajectory_name="trajectory",
                            save_video_trigger=save_train_video_trigger,
                            max_steps_per_video=max_steps_per_video,
                            video_fps=30)
    vec = ManiSkillVectorEnv(env, num_envs,
                             ignore_terminations=not partial_reset,
                             record_metrics=True)
    return vec, env


# ----------------------------- disturbance -----------------------------

def apply_disturbance(force_actor, n_envs: int, force_dim: int,
                      mag: float, prob: float, xy_only: bool, device):
    """Apply a random per-env force on `force_actor`. Mirrors single-task
    ppo.py exactly: discrete units of 0.2 in [-1, 1]^d, scaled by `mag`.

    force_dim==2: 2D force (PushCube). 3D tasks may also be forced to xy-only
    via xy_only flag."""
    if mag <= 0:
        return
    mask = (torch.rand((n_envs, 1), device=device) < prob).float()
    if force_dim == 2:
        f2 = torch.randint(-5, 6, (n_envs, 2), device=device).float() * 0.2
        f = torch.zeros((n_envs, 3), device=device)
        f[:, :2] = f2
    else:
        f = torch.randint(-5, 6, (n_envs, 3), device=device).float() * 0.2
        if xy_only:
            f[:, 2] = 0.0
    f = f * mag * mask
    force_actor.apply_force(f)


# ----------------------------- GAE -----------------------------

def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                dones: torch.Tensor, final_values: torch.Tensor,
                next_value: torch.Tensor, next_done: torch.Tensor,
                gamma: float, gae_lambda: float, finite_horizon: bool
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard CleanRL GAE, identical to ppo.py."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    lam_coef_sum = 0.0
    reward_term_sum = 0.0
    value_term_sum = 0.0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_not_done = 1.0 - next_done
            nextvalues = next_value
        else:
            next_not_done = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        real_next_values = next_not_done * nextvalues + final_values[t]
        if finite_horizon:
            lam_coef_sum = lam_coef_sum * next_not_done
            reward_term_sum = reward_term_sum * next_not_done
            value_term_sum = value_term_sum * next_not_done
            lam_coef_sum = 1 + gae_lambda * lam_coef_sum
            reward_term_sum = gae_lambda * gamma * reward_term_sum + lam_coef_sum * rewards[t]
            value_term_sum = gae_lambda * gamma * value_term_sum + gamma * real_next_values
            advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
        else:
            delta = rewards[t] + gamma * real_next_values - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
    returns = advantages + values
    return advantages, returns


# ----------------------------- logger -----------------------------

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None):
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


# ----------------------------- main -----------------------------

def main():
    args = tyro.cli(Args)

    args.batch_size_per_task = int(args.num_envs_per_task * args.num_steps)
    args.minibatch_size_per_task = int(args.batch_size_per_task // args.num_minibatches)
    # Each iteration steps args.num_steps in each of args.num_envs_per_task envs
    # across num_tasks tasks => total env steps per iter = N_tasks * B
    total_per_iter = num_tasks() * args.batch_size_per_task
    args.num_iterations = max(1, args.total_timesteps // total_per_iter)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"multitask__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(f"cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    # ---- build training & eval envs per task ----
    train_vecs: List[ManiSkillVectorEnv] = []
    eval_vecs: List[ManiSkillVectorEnv] = []
    force_actors_train = []
    force_actors_eval = []
    print("Building envs...")
    for spec in TASKS:
        print(f"  task {spec.task_id} ({spec.env_id})")
        n_train = args.num_envs_per_task if not args.evaluate else 1
        train_vec, _ = make_envs_for_task(
            spec, n_train, args.reconfiguration_freq, env_kwargs,
            partial_reset=args.partial_reset,
        )
        eval_video_dir = None
        if args.capture_video:
            eval_video_dir = f"runs/{run_name}/videos_{spec.short_name}"
        eval_vec, _ = make_envs_for_task(
            spec, args.num_eval_envs_per_task, args.eval_reconfiguration_freq,
            env_kwargs, record_video_dir=eval_video_dir,
            max_steps_per_video=args.num_eval_steps,
            record_trajectory=args.evaluate,
            partial_reset=args.eval_partial_reset,
        )
        train_vecs.append(train_vec)
        eval_vecs.append(eval_vec)
        # update obs_dim now that the env is built
        spec.probe_obs_dim(train_vec)
        # disturbance target actor (use NADE convention here)
        force_actors_train.append(getattr(train_vec._env.unwrapped, spec.force_actor_attr))
        force_actors_eval.append(getattr(eval_vec._env.unwrapped, spec.force_actor_attr))
        assert isinstance(train_vec.single_action_space, gym.spaces.Box)
        act_dim = int(np.prod(train_vec.single_action_space.shape))
        print(f"    obs_dim={spec.obs_dim}  act_dim={act_dim}  force_dim={spec.force_dim}")

    # action dim should be uniform (Panda pd_joint_delta_pos == 8)
    action_dim = int(np.prod(train_vecs[0].single_action_space.shape))
    for i, vec in enumerate(train_vecs[1:], 1):
        assert int(np.prod(vec.single_action_space.shape)) == action_dim, (
            f"task {i} has action_dim={np.prod(vec.single_action_space.shape)}, expected {action_dim}"
        )

    obs_dims = [s.obs_dim for s in TASKS]
    agent = MultiTaskAgent(obs_dims=obs_dims, action_dim=action_dim).to(device)

    # ---- warm-start experts from per-task ckpts ----
    if args.init_expert_ckpts:
        if len(args.init_expert_ckpts) != num_tasks():
            raise ValueError(
                f"--init_expert_ckpts must have {num_tasks()} entries (one per task, in "
                f"TASKS order push/pick/stack/peg). Got {len(args.init_expert_ckpts)}."
            )
        ckpt_paths = [None if (p == "" or p.lower() == "none") else p
                      for p in args.init_expert_ckpts]
        init_experts_from_per_task_ckpts(agent, ckpt_paths, device=str(device))

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        agent.load_state_dict(sd)
        print(f"loaded multi-task ckpt: {args.checkpoint}")

    def set_expert_requires_grad(agent, req: bool):
        for ex in agent.actor_moe.experts:
            for p in ex.parameters():
                p.requires_grad = req
        for ex in agent.critic_moe.experts:
            for p in ex.parameters():
                p.requires_grad = req

    # optionally freeze expert trunks initially (train only Proj_i and gate)
    experts_frozen = False
    if getattr(args, "freeze_expert_steps", 0) > 0:
        set_expert_requires_grad(agent, False)
        experts_frozen = True
        print(f"[*] Experts frozen for first {args.freeze_expert_steps} env steps")

    optimizer = optim.Adam([p for p in agent.parameters() if p.requires_grad], lr=args.learning_rate, eps=1e-5)

    # ---- logger ----
    logger = None
    if not args.evaluate:
        print("Running training")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
        )
        if args.track:
            import wandb
            config = vars(args)
            config["tasks"] = [s.env_id for s in TASKS]
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                       sync_tensorboard=False, config=config, name=run_name,
                       save_code=True, group="PPO-MT", tags=["ppo", "multitask", "moe"])
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ---- per-task rollout storage ----
    storage: List[Dict[str, torch.Tensor]] = []
    next_obs_list: List[torch.Tensor] = []
    next_done_list: List[torch.Tensor] = []
    action_lo_hi: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for tid, vec in enumerate(train_vecs):
        n = args.num_envs_per_task
        T = args.num_steps
        obs_dim = TASKS[tid].obs_dim
        storage.append(dict(
            obs=torch.zeros((T, n, obs_dim), device=device),
            actions=torch.zeros((T, n, action_dim), device=device),
            logprobs=torch.zeros((T, n), device=device),
            rewards=torch.zeros((T, n), device=device),
            dones=torch.zeros((T, n), device=device),
            values=torch.zeros((T, n), device=device),
            final_values=torch.zeros((T, n), device=device),
        ))
        no, _ = vec.reset(seed=args.seed + tid)
        next_obs_list.append(no)
        next_done_list.append(torch.zeros(n, device=device))
        lo = torch.from_numpy(vec.single_action_space.low).to(device)
        hi = torch.from_numpy(vec.single_action_space.high).to(device)
        action_lo_hi.append((lo, hi))

    eval_obs_list = [None] * num_tasks()
    max_eps_steps = {i: gym_utils.find_max_episode_steps_value(v._env) for i, v in enumerate(train_vecs)}
    print("max_episode_steps per task:", max_eps_steps)

    global_step = 0
    start_time = time.time()

    # =========================================================
    # Iteration loop
    # =========================================================
    for iteration in range(1, args.num_iterations + 1):
        print(f"\n=== Iter {iteration}/{args.num_iterations}  global_step={global_step} ===")
        agent.eval()

        # ---- eval ----
        do_eval = (iteration % args.eval_freq == 1) or args.evaluate
        if do_eval:
            print("Evaluating all tasks")
            avg_success = []
            for tid, eval_vec in enumerate(eval_vecs):
                spec = TASKS[tid]
                eval_obs, _ = eval_vec.reset()
                eval_metrics = defaultdict(list)
                num_episodes = 0
                n_eval = args.num_eval_envs_per_task
                for _ in range(args.num_eval_steps):
                    with torch.no_grad():
                        apply_disturbance(force_actors_eval[tid], n_eval, spec.force_dim,
                                          args.disturb_force_mag * spec.force_mag,
                                          args.disturb_prob,
                                          args.disturb_xy_only, device)
                        action = agent.get_action(eval_obs, tid, deterministic=True)
                        lo, hi = action_lo_hi[tid]
                        eval_obs, _, _, _, eval_infos = eval_vec.step(torch.clamp(action, lo, hi))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
                print(f"  [{spec.short_name}] {args.num_eval_steps * n_eval} steps  episodes={num_episodes}")
                for k, v in eval_metrics.items():
                    mean = torch.stack(v).float().mean()
                    if logger is not None:
                        logger.add_scalar(f"eval/{spec.short_name}/{k}", mean, global_step)
                    print(f"    eval_{spec.short_name}_{k}_mean={float(mean):.4f}")
                    if k == "success_once":
                        avg_success.append(float(mean))
            if avg_success and logger is not None:
                logger.add_scalar("eval/avg/success_once", float(np.mean(avg_success)), global_step)
                print(f"  >>> eval/avg/success_once = {np.mean(avg_success):.4f}")
            if args.evaluate:
                break

        if args.save_model and (iteration % args.eval_freq == 1):
            os.makedirs(f"runs/{run_name}", exist_ok=True)
            model_path = f"runs/{run_name}/multitask_ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"  model saved to {model_path}")

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ---- rollout (per task) ----
        rollout_time = time.time()
        for tid, vec in enumerate(train_vecs):
            spec = TASKS[tid]
            buf = storage[tid]
            buf["final_values"].zero_()
            next_obs = next_obs_list[tid]
            next_done = next_done_list[tid]
            n_env = args.num_envs_per_task
            lo, hi = action_lo_hi[tid]
            for step in range(args.num_steps):
                global_step += n_env
                buf["obs"][step] = next_obs
                buf["dones"][step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs, tid)
                    buf["values"][step] = value.flatten()
                buf["actions"][step] = action
                buf["logprobs"][step] = logprob

                apply_disturbance(force_actors_train[tid], n_env, spec.force_dim,
                                  args.disturb_force_mag * spec.force_mag,
                                  args.disturb_prob,
                                  args.disturb_xy_only, device)

                next_obs, reward, term, trunc, infos = vec.step(torch.clamp(action.detach(), lo, hi))
                next_done = torch.logical_or(term, trunc).to(torch.float32)
                buf["rewards"][step] = reward.view(-1) * args.reward_scale

                if "final_info" in infos:
                    done_mask = infos["_final_info"]
                    for k, v in infos["final_info"]["episode"].items():
                        if logger is not None:
                            logger.add_scalar(f"train/{spec.short_name}/{k}",
                                              v[done_mask].float().mean(), global_step)
                    with torch.no_grad():
                        idxs = torch.arange(n_env, device=device)[done_mask]
                        buf["final_values"][step, idxs] = agent.get_value(
                            infos["final_observation"][done_mask], tid
                        ).view(-1)
            next_obs_list[tid] = next_obs
            next_done_list[tid] = next_done
        rollout_time = time.time() - rollout_time

        # ---- GAE per task ----
        with torch.no_grad():
            per_task_flat = []
            for tid, vec in enumerate(train_vecs):
                buf = storage[tid]
                next_value = agent.get_value(next_obs_list[tid], tid).reshape(1, -1)
                adv, ret = compute_gae(
                    buf["rewards"], buf["values"], buf["dones"], buf["final_values"],
                    next_value, next_done_list[tid],
                    args.gamma, args.gae_lambda, args.finite_horizon_gae,
                )
                # flatten (T,B,*) -> (T*B,*)
                obs_dim = TASKS[tid].obs_dim
                per_task_flat.append(dict(
                    obs=buf["obs"].reshape(-1, obs_dim),
                    actions=buf["actions"].reshape(-1, action_dim),
                    logprobs=buf["logprobs"].reshape(-1),
                    advantages=adv.reshape(-1),
                    returns=ret.reshape(-1),
                    values=buf["values"].reshape(-1),
                ))

        # ---- PPO update ----
        # unfreeze experts when threshold reached
        if experts_frozen and global_step >= args.freeze_expert_steps:
            set_expert_requires_grad(agent, True)
            optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
            experts_frozen = False
            print(f"[*] Unfroze expert trunks at global_step={global_step}; optimizer rebuilt to include all params")

        agent.train()
        update_time = time.time()
        clipfracs = []
        last_pg = last_v = last_ent = last_old_kl = last_kl = last_lb = 0.0
        early_stop = False
        for epoch in range(args.update_epochs):
            task_order = list(range(num_tasks()))
            random.shuffle(task_order)
            for tid in task_order:
                f = per_task_flat[tid]
                B = f["obs"].shape[0]
                inds = np.arange(B)
                np.random.shuffle(inds)
                mb_size = args.minibatch_size_per_task
                for start in range(0, B, mb_size):
                    mb_inds = inds[start:start + mb_size]
                    mb_obs = f["obs"][mb_inds]
                    mb_act = f["actions"][mb_inds]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, tid, mb_act)
                    logratio = newlogprob - f["logprobs"][mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        old_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    if args.target_kl is not None and approx_kl > args.target_kl:
                        early_stop = True
                        break

                    mb_adv = f["advantages"][mb_inds]
                    if args.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    nv = newvalue.view(-1)
                    if args.clip_vloss:
                        vu = (nv - f["returns"][mb_inds]) ** 2
                        vc = f["values"][mb_inds] + torch.clamp(
                            nv - f["values"][mb_inds], -args.clip_coef, args.clip_coef)
                        vcl = (vc - f["returns"][mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(vu, vcl).mean()
                    else:
                        v_loss = 0.5 * ((nv - f["returns"][mb_inds]) ** 2).mean()

                        ent_loss = entropy.mean()
                        lb_loss = agent.gate_load_balance_loss()
                        # optional auxiliary task supervision: force gate logits to predict task id
                        task_loss = torch.tensor(0.0, device=mb_obs.device)
                        if args.task_loss_coef and args.task_loss_coef > 0.0:
                            gate_logits = getattr(agent, "_last_actor_gate_logits", None)
                            if gate_logits is not None:
                                targets = torch.full((gate_logits.shape[0],), tid, dtype=torch.long, device=gate_logits.device)
                                task_loss = nn.CrossEntropyLoss()(gate_logits, targets)

                            loss = pg_loss - args.ent_coef * ent_loss + v_loss * args.vf_coef \
                                + args.load_balance_coef * lb_loss + args.task_loss_coef * task_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    last_pg = pg_loss.item(); last_v = v_loss.item()
                    last_ent = ent_loss.item(); last_old_kl = old_kl.item()
                    last_kl = approx_kl.item(); last_lb = float(lb_loss.detach())
                    last_tl = float(task_loss.detach())
                if early_stop:
                    break
            if early_stop:
                break
        update_time = time.time() - update_time

        # ---- explained variance (over all tasks combined) ----
        all_v = torch.cat([f["values"] for f in per_task_flat]).cpu().numpy()
        all_r = torch.cat([f["returns"] for f in per_task_flat]).cpu().numpy()
        var_y = np.var(all_r)
        explained_var = float("nan") if var_y == 0 else 1 - np.var(all_r - all_v) / var_y

        if logger is not None:
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", last_v, global_step)
            logger.add_scalar("losses/policy_loss", last_pg, global_step)
            logger.add_scalar("losses/entropy", last_ent, global_step)
            logger.add_scalar("losses/old_approx_kl", last_old_kl, global_step)
            logger.add_scalar("losses/approx_kl", last_kl, global_step)
            logger.add_scalar("losses/clipfrac", float(np.mean(clipfracs) if clipfracs else 0.0), global_step)
            logger.add_scalar("losses/load_balance", last_lb, global_step)
            logger.add_scalar("losses/task_loss", last_tl, global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            sps = int(global_step / max(time.time() - start_time, 1e-6))
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_fps",
                              num_tasks() * args.num_envs_per_task * args.num_steps / max(rollout_time, 1e-6),
                              global_step)
        print(f"  SPS={int(global_step / max(time.time() - start_time, 1e-6))}  "
              f"pg={last_pg:.4f}  v={last_v:.4f}  kl={last_kl:.4f}  lb={last_lb:.4f}  "
              f"tl={last_tl:.4f}  rollout={rollout_time:.1f}s  update={update_time:.1f}s")

    # ---- final save ----
    if not args.evaluate and args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        model_path = f"runs/{run_name}/multitask_final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"final model saved to {model_path}")
    if logger is not None:
        logger.close()
    for vec in train_vecs + eval_vecs:
        vec.close()


if __name__ == "__main__":
    main()
