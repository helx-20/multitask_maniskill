"""Per-step data utilities for the unified multi-task criticality dataset.

`stage1_collect.py` now writes each rollout as a dict with:
    {
      "task_id":   int,                  # 0/1/2/3 == push/pick/stack/peg
      "env_id":    str,
      "force_dim": 2 or 3,
      "obs":       [obs_0, obs_1, ...],  # obs_t has shape (obs_dim_i,)
      "action":    [...],                # unused here
      "reward":    [...],
      "force":     [f_0, f_1, ...],      # each f_t has shape (3,) — unused dims are 0
      "success":   0 or 1,
    }

For the MoE classifier we want per-step (input_i, label, task_id) where
    input_i = obs_i (obs_dim_i) ⊕ force_i (force_dim_i)
    label   = 1 if the step belongs to a crash episode, else 0

Old single-task npys without a "task_id" field are assumed task_id=0
(PushCube). Old files with 3-dim force on a 2D-force task have the trailing
zero column dropped automatically.
"""

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np


def collect_npy_files(folder: str) -> List[str]:
    """Return sorted list of .npy files in `folder` (returns [] if folder missing)."""
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith(".npy"))


def _ep_task_id(ep: dict) -> int:
    return int(ep.get("task_id", 0))


def _ep_force_dim(ep: dict, fallback: int = 3) -> int:
    return int(ep.get("force_dim", fallback))


def episode_to_steps(episode: dict) -> Tuple[np.ndarray, int, int]:
    """Convert one episode dict into a (T, obs_dim_i + force_dim_i) feature
    array, plus episode_label and task_id.
    """
    obs = np.asarray(episode["obs"], dtype=np.float32)              # (T, obs_dim_i)
    if obs.shape[-1] < 48:
        pad = np.zeros((obs.shape[0], 48 - obs.shape[-1]), dtype=obs.dtype)
        obs = np.concatenate([obs, pad], axis=1)

    force = np.asarray(episode["force"], dtype=np.float32)          # (T, 3) (or (T, fd))
    fd = _ep_force_dim(episode, fallback=force.shape[-1] if force.ndim == 2 else 3)
    if force.ndim == 1:
        force = force.reshape(-1, fd)
    # Ensure force has 3 columns. If source has fewer (old files), pad with zeros.
    # Do NOT drop trailing zero columns when force_dim < 3 — keep the final zero dim.
    if force.shape[-1] < 3:
        pad_cols = 3 - force.shape[-1]
        force = np.concatenate([force, np.zeros((force.shape[0], pad_cols), dtype=force.dtype)], axis=1)
    T = min(obs.shape[0], force.shape[0])
    feats = np.concatenate([obs[:T], force[:T]], axis=1)            # (T, od+fd)
    ep_label = 0 if int(episode.get("success", 0)) == 1 else 1
    return feats, ep_label, _ep_task_id(episode)


def flatten_episodes(
    episodes: Iterable[dict],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Group per-step samples by task_id.

    Returns: {task_id: (X_i, y_i)}  with X_i shape (N_i, od_i + fd_i),
                                         y_i shape (N_i,).
    """
    bucket_X: Dict[int, List[np.ndarray]] = {}
    bucket_y: Dict[int, List[np.ndarray]] = {}
    for ep in episodes:
        feats, ep_label, tid = episode_to_steps(ep)
        if feats.shape[0] == 0:
            continue
        bucket_X.setdefault(tid, []).append(feats)
        bucket_y.setdefault(tid, []).append(
            np.full(feats.shape[0], ep_label, dtype=np.int64))

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for tid in bucket_X:
        out[tid] = (
            np.concatenate(bucket_X[tid], axis=0),
            np.concatenate(bucket_y[tid], axis=0),
        )
    return out


def load_episodes(path_or_paths):
    """Load one or more .npy files of episode dicts and concatenate."""
    if isinstance(path_or_paths, str):
        path_or_paths = [path_or_paths]
    out: List[dict] = []
    ep_num = {'0': 0, '1': 0, '2': 0, '3': 0}
    for p in path_or_paths:
        task_id = p.split('/')[-1].split('_')[1][-1]
        try:
            data = np.load(p, allow_pickle=True)
            out.extend(list(data))
            ep_num[task_id] += len(list(data))
        except Exception:
            import pickle
            with open(p, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, np.ndarray):
                out.extend(list(data))
            elif isinstance(data, list):
                out.extend(data)
            else:
                out.append(data)
    return out, ep_num
