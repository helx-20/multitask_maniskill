"""Single source of truth for the four ManiSkill tasks used by the unified
multi-task PPO + criticality pipeline.

Anything else (ppo_multitask, NADE wrapper, stage1_collect, criticality model)
should import TASKS / TaskSpec from here and never hard-code task constants.

obs_dim is intentionally left as None for some tasks — call
`TaskSpec.probe_obs_dim(env)` once you have a live env to fill it in.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TaskSpec:
    task_id: int
    env_id: str
    short_name: str
    obs_dim: Optional[int]      # state obs dim (filled at runtime if None)
    force_dim: int              # 2 for PushCube (xy only), 3 for 3D-force tasks
    xy_only: bool               # equivalent to force_dim == 2; explicit flag for readability
    force_actor_attr: str       # attribute name on env.unwrapped that receives apply_force
    stage1_force_actor_attr: str
    force_mag: float = 1.0      # absolute disturbance magnitude for this task
                                # (used by ppo_multitask, NADE wrapper, stage1_collect)
    # ↑ stage1_collect.py historically used a different actor for StackCube
    #   (cubeA) than the NADE wrapper (peg). We keep both names so that
    #   behaviour is preserved exactly. For all other tasks the two are the
    #   same. TODO: confirm with user whether StackCube's mismatch is
    #   intentional; if not, set stage1_force_actor_attr = force_actor_attr.

    @property
    def grid_size_per_dim(self) -> int:
        return 11

    @property
    def total_force_grid(self) -> int:
        return self.grid_size_per_dim ** self.force_dim

    def probe_obs_dim(self, env) -> int:
        """Reset the env once if needed and record the flat state obs dim.

        Idempotent — caches into self.obs_dim and just returns it on re-call.
        """
        if self.obs_dim is not None:
            return self.obs_dim
        # ManiSkillVectorEnv has single_observation_space
        if hasattr(env, "single_observation_space"):
            import numpy as np
            dim = int(np.prod(env.single_observation_space.shape))
        else:
            obs, _ = env.reset()
            import torch
            if torch.is_tensor(obs):
                arr = obs.detach().cpu().numpy()
            else:
                arr = obs
            dim = int(arr.shape[-1])
        self.obs_dim = dim
        return dim


# Canonical task list. Order is fixed (task_id == index) because the MoE
# layout (Proj_i, Expert_i) is indexed by task_id.
TASKS: List[TaskSpec] = [
    TaskSpec(
        task_id=0,
        env_id="PushCube-v1",
        short_name="push",
        obs_dim=35,
        force_dim=2,
        xy_only=True,
        force_actor_attr="obj",
        stage1_force_actor_attr="obj",
        force_mag=2.0,
    ),
    TaskSpec(
        task_id=1,
        env_id="PickCube-v1",
        short_name="pick",
        obs_dim=42,
        force_dim=3,
        xy_only=False,
        force_actor_attr="cube",
        stage1_force_actor_attr="cube",
        force_mag=2.0,
    ),
    TaskSpec(
        task_id=2,
        env_id="StackCube-v1",
        short_name="stack",
        obs_dim=48,
        force_dim=3,
        xy_only=False,
        # Historically there was a mismatch: some code targeted "peg" while
        # stage1_collect targeted "cubeA". The StackCube env exposes the
        # stacked cube as `cubeA`, so use that for applying disturbances.
        force_actor_attr="cubeA",
        # stage1_collect historically targets cubeA — see TaskSpec docstring.
        stage1_force_actor_attr="cubeA",
        force_mag=1.0,
    ),
    TaskSpec(
        task_id=3,
        env_id="PegInsertionSide-v1",
        short_name="peg",
        obs_dim=43,
        force_dim=3,
        xy_only=False,
        force_actor_attr="peg",
        stage1_force_actor_attr="peg",
        force_mag=6.0,
    ),
]


# ---------- lookup helpers ----------

def by_env_id(env_id: str) -> TaskSpec:
    for t in TASKS:
        if t.env_id == env_id:
            return t
    raise KeyError(f"unknown env_id: {env_id}; known: {[t.env_id for t in TASKS]}")


def by_task_id(task_id: int) -> TaskSpec:
    if not (0 <= task_id < len(TASKS)):
        raise IndexError(f"task_id {task_id} out of range [0, {len(TASKS)})")
    return TASKS[task_id]


def by_short_name(name: str) -> TaskSpec:
    for t in TASKS:
        if t.short_name == name:
            return t
    raise KeyError(f"unknown short name: {name}")


def num_tasks() -> int:
    return len(TASKS)


def obs_dims() -> List[Optional[int]]:
    return [t.obs_dim for t in TASKS]


def force_dims() -> List[int]:
    return [t.force_dim for t in TASKS]
