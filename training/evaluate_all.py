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
    print(f'Failure rate: {np.sum(crashes) / len(crashes)}')
    return np.sum(crashes) / len(crashes)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_path', default='all_results',
                    help='Path to .npy file, directory of .npy, or glob pattern for original policy results')
    args = ap.parse_args()

    short_names = ['push', 'pick', 'stack', 'peg']

    min_avg_failure_rate = float('inf')
    for dir in os.listdir(args.root_path):
        print(dir)
        data_dir = os.path.join(args.root_path, dir)
        avg_failure_rate = 0
        for s in short_names:
            print(f'=== {s} ===')
            failure_rate = analyze(data_dir, task_filter=s)
            avg_failure_rate += failure_rate
        avg_failure_rate /= len(short_names)
        if avg_failure_rate < min_avg_failure_rate:
            min_avg_failure_rate = avg_failure_rate
        print(f'Average failure rate: {avg_failure_rate}, min so far: {min_avg_failure_rate}')


if __name__ == '__main__':
    main()
