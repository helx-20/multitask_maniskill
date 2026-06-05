"""Stage-1 trajectory collection for ANY of the four ManiSkill tasks.

Driven by --task_id (0=push, 1=pick, 2=stack, 3=peg) or --env_id; both keep
the originally per-task script's behaviour 1:1:

  - obs dim is task-specific (35 / 42 / 48 / ~43; probed at runtime).
  - force vector dim is 2 for PushCube (xy only), 3 for the others.
  - The force target actor name comes from
    `task_registry.TaskSpec.stage1_force_actor_attr` (StackCube uses cubeA
    here to match the historical single-task script).
  - Episode dict now carries `task_id` so downstream training can route per
    sample. Existing npys without this field are assumed task_id=0.

Pos / neg semantics are unchanged from the original (criticality "positive"
= failed episode useful for training crash classifier).
"""

import os, sys
import pickle
import argparse
import numpy as np
import torch
import gymnasium as gym

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mani_skill.envs  # noqa: F401  (registers envs)
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from examples.baselines.ppo.multitask_agent import MultiTaskAgent
from examples.baselines.ppo.task_registry import TASKS, by_env_id, by_task_id


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sample_unit_force(force_dim: int, xy_only: bool) -> np.ndarray:
    """Discrete grid in [-1, 1] step 0.2, returns shape (3,) regardless of
    force_dim so all-task data files share the same force schema. Trailing
    dims that don't apply to the task are zero."""
    out = np.zeros(3, dtype=np.float32)
    if force_dim == 2:
        f = np.random.randint(-5, 6, size=2).astype(np.float32) * 0.2
        out[:2] = f
    else:
        f = np.random.randint(-5, 6, size=3).astype(np.float32) * 0.2
        out[:] = f
        if xy_only:
            out[2] = 0.0
    return out


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # select tasks: single task (by id/env) or all tasks
    if args.all_tasks:
        selected_specs = TASKS
    else:
        # resolve single task spec
        if args.task_id is not None:
            spec = by_task_id(args.task_id)
            if args.env_id is not None and args.env_id != spec.env_id:
                print(f"[stage1_collect][WARN] --env_id {args.env_id} disagrees with "
                      f"--task_id {args.task_id} ({spec.env_id}); using task_id.")
        elif args.env_id is not None:
            spec = by_env_id(args.env_id)
        else:
            raise ValueError("Must pass --task_id or --env_id, or use --all_tasks")
        selected_specs = [spec]

    os.makedirs(args.pos_dir, exist_ok=True)
    os.makedirs(args.neg_dir, exist_ok=True)

    env_kwargs = dict(obs_mode=args.obs_mode, sim_backend=args.sim_backend)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    # probe obs dims across selected tasks to compute global padding length
    obs_dims = {}
    for spec in selected_specs:
        env_probe = gym.make(spec.env_id, num_envs=1,
                              reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
        if isinstance(env_probe.action_space, gym.spaces.Dict):
            env_probe = FlattenActionSpaceWrapper(env_probe)
        env_probe = ManiSkillVectorEnv(env_probe, num_envs=1,
                                       ignore_terminations=not args.partial_reset,
                                       record_metrics=False)
        try:
            obs_probe, _ = env_probe.reset(seed=args.seed)
            obs_dim = int(obs_probe.shape[-1])
        finally:
            env_probe.close()
        obs_dims[spec.task_id] = obs_dim
    max_obs_dim = max(obs_dims.values())

    # iterate tasks and collect
    for spec in selected_specs:
        print(f"[stage1_collect] task {spec.task_id} = {spec.env_id} "
              f"(force_dim={spec.force_dim}, actor={spec.stage1_force_actor_attr})")

        env = gym.make(spec.env_id, num_envs=1,
                       reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        env = ManiSkillVectorEnv(env, num_envs=1,
                                 ignore_terminations=not args.partial_reset,
                                 record_metrics=True)

        action_low = torch.tensor(env.single_action_space.low, device=device, dtype=torch.float32)
        action_high = torch.tensor(env.single_action_space.high, device=device, dtype=torch.float32)

        obs_dims = [s.obs_dim for s in TASKS]
        agent = MultiTaskAgent(obs_dims=obs_dims, action_dim=8).to(device)
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        agent.eval()

        pos, neg = [], []
        for ep in range(args.n):
            obs, _ = env.reset(seed=args.seed + ep)
            base_env = env._env.unwrapped
            force_actor = getattr(base_env, spec.stage1_force_actor_attr)

            if ep == 0:
                obs_dim = int(obs.shape[-1])
                expected = spec.obs_dim
                print(f"[stage1_collect] detected state obs dim = {obs_dim}"
                      f" (registry expected {expected})")
                if expected is not None and obs_dim != expected:
                    print(f"[stage1_collect][WARN] obs dim {obs_dim} != registry {expected}")
                # update registry in-memory (for current process)
                spec.obs_dim = obs_dim

            data_episode = {
                "task_id": spec.task_id,
                "env_id": spec.env_id,
                "obs": [],
                "action": [],
                "reward": [],
                "force": [],            # always shape (3,); unused dims = 0
                "force_dim": spec.force_dim,
                "success": 0,
            }

            success_once = False
            done = False
            while not done:
                with torch.no_grad():
                    action = agent.get_action(obs.to(device), deterministic=True)

                if args.force_mag > 0 and np.random.rand() < args.force_prob:
                    f_unit = sample_unit_force(spec.force_dim, args.xy_only)
                    f_applied = f_unit * args.force_mag * spec.force_mag
                else:
                    f_unit = np.zeros(3, dtype=np.float32)
                    f_applied = np.zeros(3, dtype=np.float32)

                # apply force: cube actor expects shape (1, 3) tensor / numpy (3,)
                if args.sim_backend == "physx_cuda":
                    ft = torch.from_numpy(f_applied).to(device)
                    force_actor.apply_force(ft.view(1, 3))
                else:
                    force_actor.apply_force(f_applied.astype(np.float32))

                # pad obs with zeros to max_obs_dim so all tasks share same obs length
                o = to_np(obs[0])
                if o.shape[-1] < max_obs_dim:
                    pad = np.zeros((max_obs_dim - o.shape[-1],), dtype=o.dtype)
                    o = np.concatenate([o, pad])
                data_episode["obs"].append(o)
                data_episode["action"].append(to_np(action[0]))
                data_episode["force"].append(f_unit.copy())
                action = torch.clamp(action, action_low, action_high)
                next_obs, reward, terminated, truncated, info = env.step(action)
                data_episode["reward"].append(float(to_np(reward[0]).item()))

                if "_final_info" in info and bool(info["_final_info"][0].item()):
                    ep_metrics = info["final_info"]["episode"]
                    success_once = bool(ep_metrics["success_once"][0].item())

                obs = next_obs
                done = bool((terminated | truncated)[0].item())

            data_episode["success"] = 1 if success_once else 0
            # Same convention as the original per-task script:
            #   failed (no success this episode) -> "positive" (critical)
            #   succeeded                        -> "negative" (safe)
            if success_once:
                neg.append(data_episode)
            else:
                pos.append(data_episode)

            if (ep + 1) % args.save_interval == 0:
                _save(args, pos, neg, spec.task_id)

            if (ep + 1) % 10 == 0:
                print(f"[{ep+1}/{args.n}] task={spec.task_id} pos={len(pos)} neg={len(neg)} "
                      f"last_ep_success={int(success_once)}")

        env.close()
        _save(args, pos, neg, spec.task_id)



def _save(args, pos, neg, task_id):
    """Save per-task files: include task id to avoid collisions."""
    out_pos = os.path.join(args.pos_dir, f"pos_task{task_id}_worker{args.worker_id}.npy")
    out_neg = os.path.join(args.neg_dir, f"neg_task{task_id}_worker{args.worker_id}.npy")
    with open(out_pos, "wb") as f:
        pickle.dump(np.array(pos, dtype=object), f, protocol=4)
    with open(out_neg, "wb") as f:
        pickle.dump(np.array(neg, dtype=object), f, protocol=4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # task selection: pass either --task_id (preferred) or --env_id
    p.add_argument("--task_id", type=int, default=None,
                   help="0=push, 1=pick, 2=stack, 3=peg")
    p.add_argument("--env_id", type=str, default=None)
    p.add_argument("--all_tasks", default=True,
                   help="collect for all tasks: each worker samples each task n times")
    p.add_argument("--checkpoint", type=str, default='examples/baselines/ppo/runs/multitask__ppo_multitask__1__1780644413/multitask_final_ckpt.pt',
                   help="single-task baseline ckpt for this task")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--worker_id", type=int, default=0)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--pos_dir", type=str, default="data/stage1/positive")
    p.add_argument("--neg_dir", type=str, default="data/stage1/negative")
    p.add_argument("--obs_mode", type=str, default="state")
    p.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    p.add_argument("--sim_backend", type=str, default="physx_cpu")
    p.add_argument("--reconfiguration_freq", type=int, default=None)
    p.add_argument("--partial_reset", action="store_true", default=True)
    p.add_argument("--force_mag", type=float, default=1.0)
    p.add_argument("--force_prob", type=float, default=1.0)
    p.add_argument("--xy_only", action="store_true", default=False)
    main(p.parse_args())
