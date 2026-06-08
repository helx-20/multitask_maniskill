"""Per-task NADE wrapper, unified across the four ManiSkill tasks.

Task-specific values (obs_dim, force_dim, target actor, grid points) come
from `task_registry.TaskSpec`. The criticality model is the MoE
`MultiTaskClassifier`, which dispatches by `task_id`. Single-task
`SimpleClassifier` ckpts can still be loaded by setting
`--criticality_use_single_task` (forwards a 51-d input through the bare
SimpleClassifier).
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers envs)
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils import common

from examples.baselines.ppo.task_registry import TASKS, TaskSpec, by_env_id, by_task_id


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class NADEConfig:
    grid_size: int = 11
    xy_only: bool = False
    force_mag: float = 1.0
    force_prob: float = 1.0
    update_every: int = 1
    history_len: int = 1


def _build_force_grid(force_dim: int, grid_size: int, device) -> torch.Tensor:
    """Discrete (force_dim)-D grid in [-1, 1]; returns (G^force_dim, force_dim)."""
    vals = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=torch.float32)
    if force_dim == 2:
        fx, fy = torch.meshgrid(vals, vals, indexing="ij")
        return torch.stack([fx, fy], dim=-1).reshape(-1, 2)
    elif force_dim == 3:
        fx, fy, fz = torch.meshgrid(vals, vals, vals, indexing="ij")
        return torch.stack([fx, fy, fz], dim=-1).reshape(-1, 3)
    else:
        raise ValueError(f"unsupported force_dim={force_dim}")


class ManiSkillOrdinaryNADE(gym.Wrapper):
    def __init__(self, env, cfg: NADEConfig, args, spec: TaskSpec):
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env
        self.cfg = cfg
        self.args = args
        # `spec` is a common read-only property on gym.Env; avoid shadowing it.
        self.task_spec = spec
        self.device = torch.device(args.device)
        self.force_dim = self.task_spec.force_dim

        # ---- criticality model ----
        from criticality.utils.criticality_model import SimpleClassifier
        # SimpleClassifier expects obs_dim + 3 features (legacy collectors).
        self.criticality_model = SimpleClassifier(input_dim=51).to(self.device)

        if getattr(args, "criticality_ckpt", None):
            print(f"loading criticality ckpt from {args.criticality_ckpt}")
            ckpt = torch.load(args.criticality_ckpt, map_location=self.device)
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    sd = ckpt["model"]
                elif "state_dict" in ckpt:
                    sd = ckpt["state_dict"]
                else:
                    sd = ckpt
            else:
                sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
            self.criticality_model.load_state_dict(sd)
        self.criticality_model.eval()

        # ---- force grid for this task ----
        self.force_grid = _build_force_grid(self.force_dim, cfg.grid_size, self.device)
        self.force_grid_np = self.force_grid.detach().cpu().numpy()
        self.total_actions = self.force_grid.shape[0]

        # ---- state ----
        self.current_state = None
        self.step_count = 0
        self.total_weight = 1.0
        self.env_action = np.zeros(self.force_dim, dtype=np.float32)

        self.criticality_info = {
            "weight": 1.0,
            "p_list": np.ones(self.total_actions) / self.total_actions,
        }

        self.record_metrics = True
        self.returns = []
        self.success_once = False
        self.fail_once = False

    # ---------- env plumbing ----------
    def _get_base_env(self):
        env = self.env
        if hasattr(env, "_env"):
            env = env._env
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        return env

    def get_wrapper_attr(self, name: str):
        if hasattr(self.env, "get_wrapper_attr"):
            return self.env.get_wrapper_attr(name)
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"Underlying env has no attribute '{name}'")

    @property
    def single_observation_space(self):
        if hasattr(self.env, "single_observation_space"):
            return getattr(self.env, "single_observation_space")
        return getattr(self.env, "observation_space")

    @property
    def single_action_space(self):
        if hasattr(self.env, "single_action_space"):
            return getattr(self.env, "single_action_space")
        return getattr(self.env, "action_space")

    # ---------- criticality scoring ----------
    def _format_model_output(self, outputs: torch.Tensor) -> np.ndarray:
        if not torch.is_tensor(outputs):
            outputs = torch.as_tensor(outputs, device=self.device)
        probs = torch.softmax(outputs, dim=-1)[:, 1]
        return probs.detach().cpu().numpy()

    def _extract_state(self, obs):
        if isinstance(obs, dict):
            state = common.to_numpy(common.flatten_state_dict(obs))
        else:
            state = common.to_numpy(obs)
        if len(state.shape) > 1:
            state = state[0]
        return state

    def calcu_q(self, obs):
        """Enumerate every (force_dim)-D candidate force at the current obs."""
        cur_obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        cur_obs = cur_obs.repeat(self.total_actions, 1)  # (G^fd, obs_dim_i)
        # Legacy SimpleClassifier expected obs + 3-d force. If this is a
        # 2D-force task, pad the trailing column with zeros.
        if self.force_dim == 2:
            fg = torch.cat([self.force_grid,
                            torch.zeros(self.total_actions, 1, device=self.device)], dim=1)
        else:
            fg = self.force_grid
        cur_input = torch.cat([cur_obs, fg], dim=1)
        with torch.no_grad():
            outputs = self.criticality_model(cur_input)
        return outputs

    def idx_to_action(self, action_idx) -> np.ndarray:
        return self.force_grid_np[int(action_idx)].astype(np.float32).copy()

    def get_env_action(self, obs) -> Tuple[np.ndarray, Dict[str, Any]]:
        n = self.total_actions
        p_list = np.ones(n) / n

        if np.random.rand() > self.cfg.force_prob:
            return np.zeros(self.force_dim, dtype=np.float32), {"weight": 1.0, "p_list": p_list}

        if getattr(self.args, "nade", False):
            # Legacy collectors expected obs_dim + 3 input features; if obs_dim is smaller, pad with zeros.
            if obs.shape[-1] < 48:
                pad = np.zeros((48 - obs.shape[-1],), dtype=obs.dtype)
                obs = np.concatenate([obs, pad])
            outputs = self.calcu_q(obs)
            scores = self._format_model_output(outputs)
            criticality = scores
            if np.max(criticality) > self.args.criticality_threshold:
                alpha = 3.0
                shifted = scores - np.max(criticality)
                criticality = np.exp(shifted * alpha)
                criticality_pdf = criticality / np.sum(criticality)
                epsilon = self.args.epsilon
                pdf_array = (1 - epsilon) * criticality_pdf + epsilon * p_list
                pdf_array /= np.sum(pdf_array)
            else:
                pdf_array = p_list
        else:
            pdf_array = p_list

        pdf_array = pdf_array.astype(np.float64)
        action_idx = np.random.choice(n, p=pdf_array)
        weight = p_list[action_idx] / pdf_array[action_idx]
        return self.idx_to_action(action_idx), {"weight": weight, "p_list": p_list}

    # ---------- gym API ----------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_state = self._extract_state(obs)
        # update spec.obs_dim if it was None
        if self.task_spec.obs_dim is None:
            self.task_spec.obs_dim = int(self.current_state.shape[-1])
        self.step_count = 0
        self.total_weight = 1.0
        self.env_action = np.zeros(self.force_dim, dtype=np.float32)
        self.success_once = False
        return obs, info

    def step(self, action):
        if (self.step_count % self.cfg.update_every) == 0:
            new_env_action, criticality_info = self.get_env_action(self.current_state)
            threshold = self.args.weight_threshold
            if self.total_weight * criticality_info['weight'] < threshold:
                p_list = criticality_info['p_list']
                p_final = np.array(p_list, dtype='float64')
                p_sum = np.sum(p_final)
                p_final = np.ones_like(p_final) / len(p_final) if p_sum <= 0 else p_final / p_sum
                p_final /= np.sum(p_final)
                action_idx = np.random.choice(len(p_list), p=p_final)
                if np.random.rand() > self.cfg.force_prob:
                    self.env_action = np.zeros(self.force_dim, dtype=np.float32)
                else:
                    self.env_action = self.idx_to_action(action_idx)
                criticality_info['weight'] = 1.0
            else:
                self.env_action = new_env_action
                self.total_weight *= criticality_info['weight']
            self.criticality_info = criticality_info
            self.criticality_info['total_weight'] = self.total_weight
        else:
            self.criticality_info["weight"] = 1.0

        force = (self.env_action.detach().cpu().numpy().reshape(-1)
                 if torch.is_tensor(self.env_action)
                 else np.array(self.env_action).reshape(-1))

        # Pad force to 3D for apply_force (physics expects 3D vector).
        if force.shape[0] == 2:
            force3 = np.zeros(3, dtype=np.float32)
            force3[:2] = force
            force = force3

        ap_force = force * self.cfg.force_mag
        target_actor = getattr(self.env.unwrapped, self.task_spec.force_actor_attr)
        if self.args.sim_backend == "physx_cuda":
            ft = torch.from_numpy(ap_force).to(self.device).float()
            target_actor.apply_force(ft.reshape(1, 3))
        else:
            target_actor.apply_force(ap_force.astype(np.float32))

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_state = self._extract_state(obs)
        info['criticality_info'] = self.criticality_info
        info['nade_env_action'] = self.env_action
        self.step_count += 1
        return common.unbatch(
            common.to_numpy(obs),
            common.to_numpy(reward),
            common.to_numpy(terminated),
            common.to_numpy(truncated),
            info,
        )


def make_env(args):
    """Build NADE-wrapped env for the task implied by `args`.

    args must have: env_id OR task_id, obs_mode, sim_backend, control_mode,
    grid_size, force_mag, force_prob, update_every, weight_threshold,
    epsilon, criticality_threshold, criticality_ckpt, device, nade.
    """
    import time

    # resolve task
    if getattr(args, "task_id", None) is not None:
        spec = by_task_id(int(args.task_id))
    else:
        spec = by_env_id(args.env_id)
    print(f"NADE task: {spec.task_id} = {spec.env_id} "
          f"(force_dim={spec.force_dim}, actor={spec.force_actor_attr})")

    t0 = time.time()
    ignore_term = getattr(args, "ignore_terminations", False)
    env = gym.make(
        spec.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        render_mode=getattr(args, "render_mode", None),
        control_mode=getattr(args, "control_mode", "pd_joint_delta_pos"),
        sim_backend=args.sim_backend,
        reconfiguration_freq=None,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    print(f"env build {time.time() - t0:.2f}s")

    nade_cfg = NADEConfig(
        grid_size=args.grid_size,
        force_mag=args.force_mag * spec.force_mag,
        force_prob=args.force_prob,
        update_every=args.update_every,
        xy_only=spec.xy_only,
        history_len=getattr(args, "history_len", 1),
    )
    env = ManiSkillVectorEnv(env, num_envs=1, ignore_terminations=ignore_term,
                             record_metrics=True)
    env = ManiSkillOrdinaryNADE(env, nade_cfg, args, spec)
    return env
