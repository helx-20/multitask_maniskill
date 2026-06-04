"""Stage-1 trainer for the unified multi-task criticality model.

Loads all rollout .npy files from `<data_dir>/raw/positive/` and
`<data_dir>/raw/negative/` (collected by stage1_collect.py for any of the
four tasks; each episode dict carries `task_id`). The trainer:

  - Groups samples by task_id (4 buckets in general).
  - Performs *stratified* (per-task) train/val/test split so every task is
    represented in every split.
  - Caches the split into `<data_dir>/train_mt.pkl` / `val_mt.pkl` /
    `test_mt.pkl` after the first build.
  - Builds one DataLoader per task per split and iterates them round-robin
    per epoch, routing each batch through the right Proj_i of
    `MultiTaskClassifier`.

Backward-compat: episodes without `task_id` are assumed task_id=0. To train
in single-task mode just collect into one bucket — the same code path
still works.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from criticality.utils.multitask_criticality_model import MultiTaskClassifier
from criticality.utils.data_utils import collect_npy_files, flatten_episodes, load_episodes
from examples.baselines.ppo.task_registry import TASKS, num_tasks, by_task_id

from sklearn.metrics import auc


# ---------- metrics ----------

def precision_recall_curve(y_true, y_score, num_thresholds: int = 1000):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thr = np.linspace(1.0, 0.0, num_thresholds, endpoint=False)
    if y_true.size == 0:
        prec = np.concatenate(([1.0], np.zeros(num_thresholds)))
        rec = np.concatenate(([0.0], np.zeros(num_thresholds)))
        return prec, rec, thr
    positives = int(np.sum(y_true == 1))
    prec_list, rec_list = [], []
    for t in thr:
        preds = (y_score >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = positives - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec_list.append(p)
        rec_list.append(r)
    prec = np.concatenate(([1.0], np.array(prec_list)))
    rec = np.concatenate(([0.0], np.array(rec_list)))
    return prec, rec, thr


# ---------- data loading ----------

def _split_one_task(X, y, ratios, rng, neg_train_keep_frac):
    """Replicates the single-task script's split shape: train-time keeps
    `neg_train_keep_frac` of the negative training samples (default 0.1 — see
    original code's `int(0.1*n_train_neg)`)."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_train_pos = int(len(pos_idx) * ratios[0])
    n_val_pos = int(len(pos_idx) * ratios[1])
    n_train_neg = int(len(neg_idx) * ratios[0])
    n_val_neg = int(len(neg_idx) * ratios[1])

    tr_pos = pos_idx[:n_train_pos]
    va_pos = pos_idx[n_train_pos:n_train_pos + n_val_pos]
    te_pos = pos_idx[n_train_pos + n_val_pos:]

    tr_neg = neg_idx[:int(neg_train_keep_frac * n_train_neg)]
    va_neg = neg_idx[n_train_neg:n_train_neg + n_val_neg]
    te_neg = neg_idx[n_train_neg + n_val_neg:]

    def cat(idx_pos, idx_neg):
        return (np.concatenate([X[idx_pos], X[idx_neg]], axis=0),
                np.concatenate([y[idx_pos], y[idx_neg]], axis=0))
    return cat(tr_pos, tr_neg), cat(va_pos, va_neg), cat(te_pos, te_neg)


def build_split(data_dir: str, rng: np.random.Generator | None = None,
                ratios=(0.8, 0.1, 0.1), neg_train_keep_frac: float = 0.1):
    """Load all episodes, group by task_id, stratified per-task split.

    Returns
        train, val, test  — each is {task_id: (X_i, y_i)}.
    """
    pos_files = collect_npy_files(os.path.join(data_dir, "raw", "positive"))
    neg_files = collect_npy_files(os.path.join(data_dir, "raw", "negative"))
    print(f"[stage1] pos files: {len(pos_files)} | neg files: {len(neg_files)}")
    pos_eps = load_episodes(pos_files) if pos_files else []
    neg_eps = load_episodes(neg_files) if neg_files else []
    print(f"[stage1] pos episodes: {len(pos_eps)} | neg episodes: {len(neg_eps)}")

    per_task_pos = flatten_episodes(pos_eps)
    per_task_neg = flatten_episodes(neg_eps)
    if not per_task_pos and not per_task_neg:
        raise RuntimeError("No data found under the provided pos_dir / neg_dir")

    rng = rng or np.random.default_rng(0)
    train, val, test = {}, {}, {}
    all_tids = sorted(set(per_task_pos) | set(per_task_neg))
    for tid in all_tids:
        Xp, yp = per_task_pos.get(tid, (np.zeros((0, 1), dtype=np.float32),
                                         np.zeros((0,), dtype=np.int64)))
        Xn, yn = per_task_neg.get(tid, (np.zeros((0, 1), dtype=np.float32),
                                         np.zeros((0,), dtype=np.int64)))
        if Xp.shape[0] == 0 and Xn.shape[0] == 0:
            continue
        # Align widths (should already match — same task produces same dim)
        if Xp.shape[0] == 0:
            Xp = np.zeros((0, Xn.shape[1]), dtype=np.float32)
        if Xn.shape[0] == 0:
            Xn = np.zeros((0, Xp.shape[1]), dtype=np.float32)
        X = np.concatenate([Xp, Xn], axis=0)
        y = np.concatenate([yp, yn], axis=0)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = _split_one_task(
            X, y, ratios, rng, neg_train_keep_frac)
        try:
            short = by_task_id(tid).short_name
        except IndexError:
            short = f"task{tid}"
        print(f"  [task {tid} ({short})] dim={X.shape[1]}  "
              f"train={len(y_tr)} val={len(y_va)} test={len(y_te)}  "
              f"pos_train={int((y_tr == 1).sum())} neg_train={int((y_tr == 0).sum())}")
        train[tid] = (X_tr, y_tr)
        val[tid] = (X_va, y_va)
        test[tid] = (X_te, y_te)
    return train, val, test


# ---------- DataLoader helpers ----------

def make_loaders(per_task: dict, batch_size: int, shuffle: bool):
    """Return {task_id: DataLoader}."""
    loaders = {}
    for tid, (X, y) in per_task.items():
        if X.shape[0] == 0:
            continue
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loaders[tid] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=0, pin_memory=True)
    return loaders


def evaluate(model, loaders: dict, device):
    """Evaluate over all per-task loaders. Returns per-task and aggregate metrics."""
    model.eval()
    per_task_metrics = {}
    all_true, all_score = [], []
    total = correct = 0
    with torch.no_grad():
        for tid, loader in loaders.items():
            t_true, t_score = [], []
            t_total = t_correct = t_tp = t_fp = t_fn = 0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb, tid)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)
                t_total += yb.size(0); t_correct += (preds == yb).sum().item()
                t_tp += int(((preds == 1) & (yb == 1)).sum().item())
                t_fp += int(((preds == 1) & (yb == 0)).sum().item())
                t_fn += int(((preds == 0) & (yb == 1)).sum().item())
                t_score.extend(probs.cpu().numpy().tolist())
                t_true.extend(yb.cpu().numpy().tolist())
            acc = t_correct / t_total if t_total else 0.0
            p = t_tp / (t_tp + t_fp) if (t_tp + t_fp) else 0.0
            r = t_tp / (t_tp + t_fn) if (t_tp + t_fn) else 0.0
            prec_arr, rec_arr, thr = precision_recall_curve(t_true, t_score)
            pr_auc = auc(rec_arr, prec_arr)
            per_task_metrics[tid] = dict(acc=acc, precision=p, recall=r, auc=pr_auc,
                                         prec_arr=prec_arr, rec_arr=rec_arr, thr=thr)
            all_true.extend(t_true); all_score.extend(t_score)
            total += t_total; correct += t_correct
    prec_arr, rec_arr, thr = precision_recall_curve(all_true, all_score)
    pr_auc = auc(rec_arr, prec_arr)
    agg = dict(acc=correct / max(total, 1), auc=pr_auc,
               prec_arr=prec_arr, rec_arr=rec_arr, thr=thr)
    return per_task_metrics, agg


def save_pr_curve(metrics: dict, path: str, title_extra: str = ""):
    if plt is None:
        return
    prec, rec, thr = metrics["prec_arr"], metrics["rec_arr"], metrics["thr"]
    if prec.size == 0:
        return
    fig = plt.figure()
    plt.step(rec, prec, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall {title_extra} (AUC={metrics['auc']:.4f})")
    plt.grid(True); plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


# ---------- train / test ----------

def _cache_paths(data_dir):
    return (os.path.join(data_dir, "train_mt.pkl"),
            os.path.join(data_dir, "val_mt.pkl"),
            os.path.join(data_dir, "test_mt.pkl"))


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    tr_path, va_path, te_path = _cache_paths(args.data_dir)
    if all(os.path.exists(p) for p in (tr_path, va_path, te_path)) and not args.rebuild:
        with open(tr_path, "rb") as f: train_dict = pickle.load(f)
        with open(va_path, "rb") as f: val_dict = pickle.load(f)
        with open(te_path, "rb") as f: test_dict = pickle.load(f)
        print(f"[stage1] loaded cached splits from {args.data_dir}")
    else:
        train_dict, val_dict, test_dict = build_split(args.data_dir, rng=rng)
        with open(tr_path, "wb") as f: pickle.dump(train_dict, f, protocol=4)
        with open(va_path, "wb") as f: pickle.dump(val_dict, f, protocol=4)
        with open(te_path, "wb") as f: pickle.dump(test_dict, f, protocol=4)

    # Build per-task loaders
    train_loaders = make_loaders(train_dict, args.batch_size, shuffle=True)
    val_loaders = make_loaders(val_dict, args.batch_size, shuffle=False)
    test_loaders = make_loaders(test_dict, args.batch_size, shuffle=False)
    present_tids = sorted(train_loaders.keys())
    print(f"[stage1] task_ids present in train: {present_tids}")

    # Build model. We use the full registry's obs/force dims so model layout
    # is identical regardless of which tasks are in this run's data — keeps
    # the ckpt portable.
    obs_dims = []
    force_dims = []
    for spec in TASKS:
        od = spec.obs_dim
        if od is None and spec.task_id in train_dict:
            # infer from data column width
            od = train_dict[spec.task_id][0].shape[1] - spec.force_dim
        obs_dims.append(od or 1)
        force_dims.append(spec.force_dim)
    print(f"[stage1] model obs_dims={obs_dims} force_dims={force_dims}")

    model = MultiTaskClassifier(obs_dims=obs_dims, force_dims=force_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"multitask_criticality_best_{args.model_idx}.pt")

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = 0
        # interleave tasks: zip the longest, looping shorter ones — keeps
        # gradient updates mixed across tasks throughout the epoch.
        iters = {tid: iter(ld) for tid, ld in train_loaders.items()}
        max_len = max((len(ld) for ld in train_loaders.values()), default=0)
        for step_idx in range(max_len):
            tid_order = list(iters.keys()); np.random.shuffle(tid_order)
            for tid in tid_order:
                try:
                    xb, yb = next(iters[tid])
                except StopIteration:
                    iters[tid] = iter(train_loaders[tid])
                    xb, yb = next(iters[tid])
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb, tid)
                loss = criterion(logits, yb)
                if args.load_balance_coef > 0:
                    loss = loss + args.load_balance_coef * model.gate_load_balance_loss()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                preds = logits.argmax(dim=1)
                total += yb.size(0); correct += (preds == yb).sum().item()
        train_acc = correct / max(total, 1)

        per_task_val, agg_val = evaluate(model, val_loaders, device)
        log_line = f"Epoch {epoch:3d}/{args.epochs} | train_acc={train_acc:.4f} | val_acc={agg_val['acc']:.4f} val_auc={agg_val['auc']:.4f}"
        for tid in present_tids:
            if tid in per_task_val:
                m = per_task_val[tid]
                log_line += f" | t{tid}:auc={m['auc']:.3f}"
        print(log_line)

        if agg_val["auc"] >= best_auc:
            best_auc = agg_val["auc"]
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best ckpt (auc={best_auc:.4f}) to {save_path}")

    # Final test
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    per_task_test, agg_test = evaluate(model, test_loaders, device)
    print(f"\n[stage1][TEST] aggregate acc={agg_test['acc']:.4f} auc={agg_test['auc']:.4f}")
    save_pr_curve(agg_test,
                  os.path.join(args.save_dir, f"precision_recall_all_{args.model_idx}.png"),
                  title_extra="(all tasks)")
    for tid, m in per_task_test.items():
        short = by_task_id(tid).short_name
        print(f"  task {tid} ({short}): acc={m['acc']:.4f} p={m['precision']:.4f} "
              f"r={m['recall']:.4f} auc={m['auc']:.4f}")
        save_pr_curve(m, os.path.join(args.save_dir,
                                       f"precision_recall_{short}_{args.model_idx}.png"),
                      title_extra=f"({short})")


def test_only(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _, _, te_path = _cache_paths(args.data_dir)
    if not os.path.exists(te_path):
        print(f"[stage1][TEST] no cached test split at {te_path}; run train first")
        return
    with open(te_path, "rb") as f:
        test_dict = pickle.load(f)
    test_loaders = make_loaders(test_dict, args.batch_size, shuffle=False)

    obs_dims = [s.obs_dim or 1 for s in TASKS]
    force_dims = [s.force_dim for s in TASKS]
    model = MultiTaskClassifier(obs_dims=obs_dims, force_dims=force_dims).to(device)
    ckpt = os.path.join(args.save_dir, f"multitask_criticality_best_{args.model_idx}.pt")
    if not os.path.exists(ckpt):
        print(f"[stage1][TEST] no ckpt at {ckpt}; abort")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device))
    per_task_test, agg_test = evaluate(model, test_loaders, device)
    print(f"[stage1][TEST] aggregate acc={agg_test['acc']:.4f} auc={agg_test['auc']:.4f}")
    for tid, m in per_task_test.items():
        short = by_task_id(tid).short_name
        print(f"  task {tid} ({short}): acc={m['acc']:.4f} p={m['precision']:.4f} "
              f"r={m['recall']:.4f} auc={m['auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root containing raw/positive and raw/negative subdirs")
    parser.add_argument("--save_dir", type=str, default="criticality/stage1/model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_idx", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_balance_coef", type=float, default=0.01)
    parser.add_argument("--rebuild", action="store_true",
                        help="Ignore cached splits and rebuild from raw npys")
    parser.add_argument("--test", action="store_true",
                        help="Only run evaluation on the best ckpt")
    args = parser.parse_args()
    print("args:", args)
    if args.test:
        test_only(args)
    else:
        train(args)
