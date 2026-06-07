#!/usr/bin/env python3
"""Paired significance test for two test_model.py evaluation runs.

Assumes both runs used identical worker_id/seed sequences so episodes are
paired element-wise across the two result arrays.

Auto-detects output type:
  - binary (NDE, crash in {0, 1})           -> McNemar exact + Newcombe paired CI
  - weighted (NADE, crash = 0 or weight>0)  -> paired t-test + paired bootstrap CI

Usage:
    # single file per policy
    python training/evaluate.py --orig results/orig/nde_0.npy --new results/new/nde_0.npy

    # directory (concat all *.npy inside) or glob pattern
    python training/evaluate.py --orig results/orig --new results/new
    python training/evaluate.py --orig 'results/orig/nde_*.npy' --new 'results/new/nde_*.npy'
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import sys
import numpy as np

try:
    from scipy.stats import binomtest, norm, ttest_rel
except ImportError:
    print('Requires scipy. Install with: pip install scipy')
    sys.exit(1)

import numpy as np
import os
from scipy.stats import norm
import math

alpha = 0.05
z = norm.isf(q=alpha)

def calculate_val(the_list):
    Mean = []
    Relative_half_width = []
    Var = []
    var_old = 0
    mean_old = 0
    for i in range(len(the_list)):
        if math.isnan(the_list[i]) or math.isinf(the_list[i]):
            the_list[i] = 0.0
        n = i + 1
        mean_new = mean_old + (the_list[i] - mean_old) / n
        Mean.append(mean_new)
        var_new = (n - 1) * var_old / n + (n - 1) * (the_list[i] - mean_old) ** 2 / (n * n)
        Var.append(1.96 * (np.sqrt(var_new / n)))
        Relative_half_width.append(z * (np.sqrt(var_new / n) / (mean_new + 1e-30)))
        var_old = var_new
        mean_old = mean_new
    return Mean, Relative_half_width, Var

def analyze(path, task_filter: str | None = None):
    crashes = []
    if os.path.isfile(path):
        files = [os.path.basename(path)]
        path = os.path.dirname(path) or "."
    else:
        files = os.listdir(path)
    for file in files:
        if task_filter and not (f"_{task_filter}_" in file or file.startswith(f"{task_filter}_")):
            continue
        try:
            data = np.load(os.path.join(path, file), allow_pickle=True).tolist()
            # print([data[i] for i in range(len(data)) if data[i] > 0])
            crashes.extend(data)
        except:
            continue
    # np.save("/home/linxuan/Embodied/go2_mujoco/results/nade_all.npy", np.array(crashes[:200000]))
    mean, rhf, var = calculate_val(crashes)
    print(f'Failure rate: {np.sum(crashes) / len(crashes)}')
    print(f'Mean: {mean[-1]:.6f}, Relative Half Width: {rhf[-1]:.6f}, Variance: {var[-1]:.6f}')
    print(f'Total samples: {len(crashes)}, Num of crashes: {np.sum(np.array(crashes) > 0)}, Max weight: {np.max(crashes)} \n')


def resolve_files(pattern: str, task_filter: str | None = None) -> list[str]:
    if os.path.isdir(pattern):
        files = sorted(glob.glob(os.path.join(pattern, '*.npy')))
    elif os.path.isfile(pattern):
        files = [pattern]
    else:
        files = sorted(glob.glob(pattern))
    if task_filter:
        # filter by task short_name token in filename (nade_<short>_<wid>.npy)
        files = [f for f in files if f"_{task_filter}_" in os.path.basename(f)
                 or os.path.basename(f).startswith(f"{task_filter}_")]
    if not files:
        raise FileNotFoundError(f'no .npy files matched: {pattern}'
                                 + (f' (task_filter={task_filter})' if task_filter else ''))
    return files


def load_paired(pattern_orig: str, pattern_new: str,
                task_filter: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load .npy files from both sides, paired by sorted order.

    - If one side has more files than the other, the extras on the longer side
      are dropped (paired prefix only).
    - Within each file pair, if episode counts differ, both are truncated to
      the shorter length.
    """
    files_o = resolve_files(pattern_orig, task_filter)
    files_n = resolve_files(pattern_new, task_filter)
    n_pair = min(len(files_o), len(files_n))
    # if len(files_o) != len(files_n):
    #     print(f'  file-count mismatch: orig={len(files_o)}, new={len(files_n)}. '
    #           f'Using first {n_pair} pair(s) in sorted order; extras dropped.')
    #     for f in files_o[n_pair:]:
    #         print(f'    drop (orig): {os.path.basename(f)}')
    #     for f in files_n[n_pair:]:
    #         print(f'    drop (new):  {os.path.basename(f)}')

    arrs_o, arrs_n = [], []
    for fo, fn in zip(files_o[:n_pair], files_n[:n_pair]):
        ao = np.load(fo, allow_pickle=False).astype(np.float64).reshape(-1)
        an = np.load(fn, allow_pickle=False).astype(np.float64).reshape(-1)
        m = min(len(ao), len(an))
        note = ''
        # if len(ao) != len(an):
        #     note = f'  (truncated from orig={len(ao)}, new={len(an)})'
        arrs_o.append(ao[:m])
        arrs_n.append(an[:m])
        # print(f'    [{os.path.basename(fo):25s} | {os.path.basename(fn):25s}]  n={m}{note}')

    out_o = np.concatenate(arrs_o) if arrs_o else np.zeros(0)
    out_n = np.concatenate(arrs_n) if arrs_n else np.zeros(0)
    print(f'  -> paired total: {len(out_o)} episodes from {n_pair} file pair(s)')
    return out_o, out_n


def is_binary(a: np.ndarray, atol: float = 1e-9) -> bool:
    """True if array values are all 0 or 1."""
    if a.size == 0:
        return True
    return bool(np.all((np.abs(a) < atol) | (np.abs(a - 1.0) < atol)))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI for a proportion, used as building block for Newcombe."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def mcnemar_analysis(orig: np.ndarray, new: np.ndarray) -> None:
    """Paired binary comparison: McNemar exact + Newcombe paired-difference CI."""
    N = len(orig)
    o = orig.astype(bool)
    n = new.astype(bool)
    a = int(np.sum(~o & ~n))
    b = int(np.sum(~o &  n))   # new crashed, orig did not  -> "new worse"
    c = int(np.sum( o & ~n))   # orig crashed, new did not  -> "new better"
    d = int(np.sum( o &  n))

    p_orig = (c + d) / N
    p_new  = (b + d) / N
    diff = p_new - p_orig

    print('\n=== 2x2 contingency (paired) ===')
    print('                  new=ok        new=crash')
    print(f'  orig=ok      {a:10d}    {b:10d}')
    print(f'  orig=crash   {c:10d}    {d:10d}')
    print(f'\n  orig crashes = {c + d:6d}   rate = {p_orig:.3e}')
    print(f'  new  crashes = {b + d:6d}   rate = {p_new :.3e}')
    print(f'  paired diff (new - orig) = {diff:+.3e}')
    print(f'  discordant  (b + c)      = {b + c}')

    if b + c == 0:
        print('\n  all pairs concordant -- no test possible (policies gave identical verdicts)')
        return

    # Exact McNemar via binomial on min(b, c) ~ Binom(b+c, 0.5)
    p_exact = binomtest(min(b, c), n=b + c, p=0.5, alternative='two-sided').pvalue
    # Chi-square with continuity correction (for reference)
    chi2_cc = (abs(b - c) - 1) ** 2 / (b + c)
    z_cc = math.sqrt(chi2_cc) if chi2_cc > 0 else 0.0
    p_chi2 = 2 * (1 - norm.cdf(z_cc))

    print('\n=== McNemar tests (H0: b = c, i.e. no difference) ===')
    print(f'  exact binomial two-sided p = {p_exact:.4f}')
    print(f'  chi2 (cc) two-sided p      = {p_chi2:.4f}')

    # Newcombe method 10 paired-difference CI
    l1, u1 = wilson_ci(c + d, N)
    l2, u2 = wilson_ci(b + d, N)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = ((b * c) - (a * d)) / math.sqrt(denom) if denom > 0 else 0.0
    t_lo = (p_orig - l1) ** 2 - 2 * phi * (p_orig - l1) * (u2 - p_new) + (u2 - p_new) ** 2
    t_hi = (u1 - p_orig) ** 2 - 2 * phi * (u1 - p_orig) * (p_new - l2) + (p_new - l2) ** 2
    lower = diff - math.sqrt(max(t_lo, 0.0))
    upper = diff + math.sqrt(max(t_hi, 0.0))
    print(f'  95% CI on paired diff: [{lower:+.3e}, {upper:+.3e}]  (Newcombe)')

    print('\n=== Verdict ===')
    alpha = 0.05
    if p_exact < alpha:
        direction = 'WORSE' if b > c else 'BETTER'
        print(f'  new policy is {direction} than orig at alpha={alpha} (p={p_exact:.4f})')
    else:
        print(f'  no significant difference at alpha={alpha} (p={p_exact:.4f})')


def weighted_analysis(orig: np.ndarray, new: np.ndarray,
                      n_boot: int = 10000, rng_seed: int = 0) -> None:
    """Paired weighted comparison: paired t-test + paired bootstrap CI."""
    N = len(orig)
    diff = new - orig
    m_orig = float(orig.mean())
    m_new = float(new.mean())
    m_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if N > 1 else 0.0

    print('\n=== Weighted crash rate (paired) ===')
    print(f'  orig:   mean = {m_orig:.3e}   sum = {orig.sum():.3f}   nonzero = {(orig > 0).sum()}')
    print(f'  new :   mean = {m_new :.3e}   sum = {new .sum():.3f}   nonzero = {(new  > 0).sum()}')
    print(f'  paired diff (new - orig):  mean = {m_diff:+.3e}   std = {sd_diff:.3e}')

    # Paired t-test (valid asymptotically even for heavy-tailed paired diffs if N large)
    t_stat, p_t = ttest_rel(new, orig)
    se = sd_diff / math.sqrt(N) if N > 0 else float('nan')
    gauss_lo = m_diff - 1.96 * se
    gauss_hi = m_diff + 1.96 * se
    print('\n=== Paired t-test ===')
    print(f'  t = {t_stat:.3f},  two-sided p = {p_t:.4f}')
    print(f'  95% CI on diff (Gaussian SE): [{gauss_lo:+.3e}, {gauss_hi:+.3e}]')

    # Paired bootstrap (resample episode indices with replacement, recompute mean diff)
    rng = np.random.default_rng(rng_seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[i] = diff[idx].mean()
    lo_b, hi_b = np.quantile(boots, [0.025, 0.975])
    # Bootstrap two-sided p via percentile of 0 under the resample distribution
    center = boots.mean()
    p_boot = 2.0 * min(float(np.mean(boots >= 0)), float(np.mean(boots <= 0)))
    print(f'\n=== Paired bootstrap (B = {n_boot}) ===')
    print(f'  95% CI on diff: [{lo_b:+.3e}, {hi_b:+.3e}]')
    print(f'  two-sided p (percentile) = {p_boot:.4f}   (bootstrap mean = {center:+.3e})')

    print('\n=== Verdict ===')
    alpha = 0.05
    # Prefer bootstrap for heavy-tailed weighted data; fall back to t if bootstrap degenerate.
    p_use = p_boot if n_boot > 0 else p_t
    if p_use < alpha:
        direction = 'WORSE' if m_diff > 0 else 'BETTER'
        print(f'  new policy is {direction} than orig at alpha={alpha} (p={p_use:.4f})')
    else:
        print(f'  no significant difference at alpha={alpha} (p={p_use:.4f})')

    # Helpful hint if paired diff is dominated by a handful of large weights
    nz = diff[diff != 0]
    if nz.size > 0 and nz.size < 20:
        print(f'\n  note: only {nz.size} episodes contribute nonzero paired diff. '
              f'Power is limited; consider more episodes or matched NADE weighting.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', default='all_results/results_origin',
                    help='Path to .npy file, directory of .npy, or glob pattern for original policy results')
    ap.add_argument('--new', default='all_results/results_origin',
                    help='Same as --orig but for the new policy')
    ap.add_argument('--mode', choices=['auto', 'binary', 'weighted'], default='auto',
                    help='Force a test mode instead of auto-detecting from values')
    ap.add_argument('--task', type=str, default=None,
                    help='Only consider files whose basename contains this task short_name '
                         '(push / pick / stack / peg). Useful when results dir holds mixed tasks.')
    ap.add_argument('--n_boot', type=int, default=10000,
                    help='Bootstrap iterations (weighted mode only)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    short_names = ['push', 'pick', 'stack', 'peg']

    # If task specified, only run that one; otherwise auto-detect tasks present in either side
    tasks_to_run = [args.task] if args.task else []
    if not tasks_to_run:
        present = set()
        for s in short_names:
            try:
                _ = resolve_files(args.orig, task_filter=s)
                present.add(s)
            except FileNotFoundError:
                pass
            try:
                _ = resolve_files(args.new, task_filter=s)
                present.add(s)
            except FileNotFoundError:
                pass
        tasks_to_run = sorted(present)

    if not tasks_to_run:
        # fallback: analyze whole dirs/files (legacy behavior)
        print(f'[load] orig: {args.orig}')
        analyze(args.orig, task_filter=None)
        print(f'[load] new : {args.new}')
        analyze(args.new, task_filter=None)
        orig, new = load_paired(args.orig, args.new, task_filter=None)
        tasks_to_run = [None]

    for task in tasks_to_run:
        print('\n' + '=' * 60)
        print(f"Analyzing task: {task if task is not None else 'ALL'}")
        print(f'[load] orig: {args.orig}' + (f'  (task={task})' if task else ''))
        analyze(args.orig, task_filter=task)
        print(f'[load] new : {args.new}' + (f'  (task={task})' if task else ''))
        analyze(args.new, task_filter=task)

        orig, new = load_paired(args.orig, args.new, task_filter=task)
        if len(orig) == 0:
            print('\nERROR: no paired episodes to analyze for this task.')
            continue

        N = len(orig)
        print(f'\n[paired] task={task if task else "ALL"} N = {N} episodes')

        if args.mode == 'auto':
            mode = 'binary' if (is_binary(orig) and is_binary(new)) else 'weighted'
        else:
            mode = args.mode
        print(f'[mode] {mode}  (--mode {args.mode})')

        if mode == 'binary':
            mcnemar_analysis(orig, new)
        else:
            weighted_analysis(orig, new, n_boot=args.n_boot, rng_seed=args.seed)


if __name__ == '__main__':
    main()
