# Multi-Task ManiSkill — Criticality/NADE Pipeline

**Software Overview**
- This software implements a ManiSkill-based "criticality/NADE" evaluation and offline retraining pipeline across 2 tabletop manipulation tasks. Main modules: multi-task MoE PPO baseline, Stage-1 criticality classifier, NADE testing/data collection, offline PPO retraining with importance sampling.

**Supported Tasks**
| task_id | short_name | env_id | obs_dim | force_dim | actor |
|---------|------------|--------|---------|-----------|-------|
| 0 | stack | StackCube-v1 | 48 | 3 | cubeA |
| 1 | peg | PegInsertionSide-v1 | 43 | 3 | peg |

Task definitions live in [`examples/baselines/ppo/task_registry.py`](examples/baselines/ppo/task_registry.py) — the single source of truth imported by all modules.

---

**Prerequisites**
- Ensure the Python environment has required dependencies installed (ManiSkill, PyTorch, scipy, etc.).
- Choose `--sim_backend physx_cpu` (numpy force API) or `physx_cuda` (torch force API) as needed. Scripts default to `physx_cpu`.

---

## 1. Multi-Task PPO Baseline

**Entry script:** [`examples/baselines/ppo/ppo_multitask.py`](examples/baselines/ppo/ppo_multitask.py)

Trains a Mixture-of-Experts (MoE) actor-critic (`MultiTaskAgent`) across 2 parallel ManiSkillVectorEnv instances. Each task gets its own expert trunk; the gate routes observations to the correct expert.

**Training:**
```bash
python examples/baselines/ppo/ppo_multitask.py \
  --num_envs_per_task=512 --num_eval_envs_per_task=32 \
  --total_timesteps=100000000 --eval_freq=10 --num_steps=100
```

**Evaluation:**
```bash
python examples/baselines/ppo/ppo_multitask.py \
  --evaluate --checkpoint path/to/multitask_final_ckpt.pt \
  --num_eval_envs_per_task=32 --num_eval_steps=1000
```

**Warm-start from single-task checkpoints:**
```bash
python examples/baselines/ppo/ppo_multitask.py \
  --init_expert_ckpts stack.pt peg.pt
```

---

## 2. Stage-1: Criticality Classifier

**Collection script:** [`criticality/stage1/stage1_collect.py`](criticality/stage1/stage1_collect.py)
**Training script:** [`criticality/stage1/stage1_train.py`](criticality/stage1/stage1_train.py)
**Model:** [`criticality/utils/criticality_model.py`](criticality/utils/criticality_model.py) (MoE `MultiTaskClassifier` dispatching by `task_id`)

**Collect positive/negative samples (per worker):**
```bash
python criticality/stage1/stage1_collect.py \
  --task_id 0 \
  --checkpoint examples/baselines/ppo/runs/<run>/multitask_final_ckpt.pt \
  --n 2000 --worker_id 0 \
  --pos_dir data/stage1/raw/positive --neg_dir data/stage1/raw/negative
```

**Train the classifier:**
```bash
python criticality/stage1/stage1_train.py \
  --data_dir /path/to/data/stage1 \
  --save_dir criticality/stage1/model --model_idx 1
```

The trainer performs stratified (per-task) train/val/test split and caches the splits as `.pkl` files under `data_dir`.

---

## 3. NADE Testing & Trajectory Collection

**Entry scripts:**
- [`criticality/test/test_model.py`](criticality/test/test_model.py) — main test harness (NDE & NADE modes, buffer collection)
- [`criticality/test/evaluate.py`](criticality/test/evaluate.py) — paired significance testing (McNemar for binary, paired t-test for weighted)
- [`criticality/test/generate_video.py`](criticality/test/generate_video.py) — render rollout videos
- [`criticality/test/maniskill_ordinary_nade.py`](criticality/test/maniskill_ordinary_nade.py) — per-task NADE wrapper (imported by test_model)

**Natural environment (uniform sampling / NDE):**
```bash
python criticality/test/test_model.py --worker_id 0 --n 200 \
  --checkpoint <policy.pt> --save_dir ./test_results
```

**Adversarial environment (NADE, criticality-based sampling):**
```bash
python criticality/test/test_model.py --worker_id 0 --n 200 --nade \
  --checkpoint <policy.pt> \
  --criticality_ckpt criticality/stage1/model/stage1_criticality_best_1.pt \
  --save_dir ./test_results
```

**Collect training buffers for offline PPO:**
```bash
python criticality/test/test_model.py --nade --training_out ./buffers/roundN \
  --checkpoint <policy.pt> --criticality_ckpt <stage1.pt>
```
Each worker produces one `training_<short>_<wid>.npy` file containing obs, actions, weights, rewards, dones, and log_probs.

**Paired significance test (compare two policies):**
```bash
python criticality/test/evaluate.py --orig results/orig --new results/new
```

---

## 4. Offline PPO Retraining

**Entry script:** [`training/ppo_offline.py`](training/ppo_offline.py)

Fine-tunes a `MultiTaskAgent` with PPO-clip + BC anchor + value loss, using importance weights from the NADE sampler. Supports multi-round iterative retraining.

**Training:**
```bash
python training/ppo_offline.py \
  --dataset /path/to/buffers/round5 \
  --initial_ckpt training/models/round4/offline_model_best.pt \
  --out_dir ./training/models/round5 \
  --epochs 50 --batch_size 1024 --learning_rate 1e-5 \
  --bc_coef 1.0 --vf_coef 1.0 --warmup_epochs 0
```

**Key arguments:**
| Argument | Description |
|----------|-------------|
| `--dataset` | One or more directories of `training_<short>_<wid>.npy` files |
| `--initial_ckpt` | Starting MultiTaskAgent checkpoint |
| `--out_dir` | Output directory for model checkpoints |
| `--bc_coef` | BC anchor loss coefficient (MSE to data actions) |
| `--vf_coef` | Value function loss coefficient |
| `--warmup_epochs` | Number of initial epochs with value-only loss (×10) |
| `--freeze_gate` | Freeze MoE gate parameters during training |
| `--task_loss_weights` | Per-task loss weights (normalized to mean=1.0) |
| `--log_std` | Fixed policy log_std (None = learned) |

On first load, the script aggregates all `.npy` files per task into `all_data_unified_weight_<short>.npy` caches; delete them to force a rebuild.

**Iterative retraining round model:**
```
training/models/
├── round1/ ... round7/    # successive offline retraining rounds
└── random/                # random-policy baseline
```

---

## 5. Evaluation & Visualization

**Batch evaluate all result directories:**
```bash
python training/evaluate_all.py --root_path all_results
```

**Plot failure rates across rounds:**
```bash
python training/draw_failure_rates.py
```

---

## Parallelization & Workers

`stage1_collect.py` and `test_model.py` support multi-process parallelism — pass different `--worker_id` values with corresponding random `--seed`. Each worker writes its output to the shared target directory (1 `.npy` per worker).

---

## Common Notes & Conventions

- **Backend:** Scripts default to `--sim_backend physx_cpu`. Under `physx_cuda`, `apply_force` requires Torch tensors; under `physx_cpu`, numpy arrays are used. The code contains both paths — maintain both when modifying.
- **Force semantics:** Unit force vectors are stored and used by the classifier (each component ∈ {-1, -0.8, ..., 1.0}), multiplied by `force_mag` when applied. All remaining tasks use 3D forces.
- **Crash labels:** If an episode never triggers `success_once`, it is labeled as a positive sample (crash=1) for the criticality classifier.
- **Observation dimension:** All modules expect exactly `obs_dim=48` (padded with zeros if the task's native dim is smaller). The MoE agent input is hardcoded to 48; individual task obs_dims are defined in `task_registry.py`.
- **Importance weights:** The NADE wrapper writes per-step `weight` and per-episode `total_weight` into `info['criticality_info']`. The offline training and evaluation scripts depend on this field — do not break this contract.
- **Task routing:** All modules use `task_registry.TaskSpec.task_id` (0–1) to route samples to the correct expert. Episode dicts carry `task_id`; legacy files without it are assumed `task_id=0`.

---

## Troubleshooting

- If the `mani_skill` package cannot be found, ensure you're using the repository root as CWD, or manually add the `mani_skill` directory to `PYTHONPATH`.
- If checkpoint path defaults don't apply (Windows/different disk), explicitly pass `--checkpoint`, `--data_dir`, or `--dataset`.
- Observation dimension mismatch warnings from `stage1_collect.py` mean the env's current obs space doesn't match expectations — update `TaskSpec.obs_dim` in the registry if the observation mode changed.

---

## File Map

| Module | Key Files |
|--------|-----------|
| Task registry | [`examples/baselines/ppo/task_registry.py`](examples/baselines/ppo/task_registry.py) |
| MoE agent | [`examples/baselines/ppo/multitask_agent.py`](examples/baselines/ppo/multitask_agent.py) |
| Multi-task PPO | [`examples/baselines/ppo/ppo_multitask.py`](examples/baselines/ppo/ppo_multitask.py) |
| Stage-1 collect | [`criticality/stage1/stage1_collect.py`](criticality/stage1/stage1_collect.py) |
| Stage-1 train | [`criticality/stage1/stage1_train.py`](criticality/stage1/stage1_train.py) |
| Criticality model | [`criticality/utils/criticality_model.py`](criticality/utils/criticality_model.py) |
| NADE wrapper | [`criticality/test/maniskill_ordinary_nade.py`](criticality/test/maniskill_ordinary_nade.py) |
| NADE testing | [`criticality/test/test_model.py`](criticality/test/test_model.py) |
| Paired eval | [`criticality/test/evaluate.py`](criticality/test/evaluate.py) |
| Video rendering | [`criticality/test/generate_video.py`](criticality/test/generate_video.py) |
| Offline PPO | [`training/ppo_offline.py`](training/ppo_offline.py) |
| Batch evaluate | [`training/evaluate_all.py`](training/evaluate_all.py) |
| Failure rate plot | [`training/draw_failure_rates.py`](training/draw_failure_rates.py) |
