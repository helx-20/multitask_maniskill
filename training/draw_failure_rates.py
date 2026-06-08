import os
import glob
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def _load_sequence(path: str) -> np.ndarray:
	"""按 `evaluate.py` 的方式加载数据：
	- 如果是目录：合并该目录下所有 `.npy` 文件（按文件名排序），使用 `allow_pickle=True`。
	- 如果是文件：直接 `np.load(..., allow_pickle=True).tolist()`。
	- 否则当成 glob pattern 使用 `glob.glob` 并按排序加载匹配到的文件。
	返回一维 numpy 数组（np.asarray）。
	"""
	def load_npy_file(fpath: str):
		data = np.load(fpath, allow_pickle=True)
		# 如果是 numpy array，尝试转为 list
		try:
			return data.tolist()
		except Exception:
			return np.asarray(data).ravel().tolist()

	items = []
	if os.path.isdir(path):
		files = sorted(glob.glob(os.path.join(path, '*.npy')))
		for f in files:
			items.extend(load_npy_file(f))
		return np.asarray(items)

	if os.path.isfile(path):
		items = load_npy_file(path)
		return np.asarray(items)

	# treat as glob pattern
	matches = sorted(glob.glob(path))
	if matches:
		for f in matches:
			if os.path.isdir(f):
				# recursively load directory
				items.extend(_load_sequence(f).tolist())
			else:
				items.extend(load_npy_file(f))
		return np.asarray(items)

	# try common extensions
	for ext in ('.npy',):
		if os.path.exists(path + ext):
			return _load_sequence(path + ext)

	raise FileNotFoundError(f"Cannot find data for path: {path}")


def plot_failure_rates(paths: List[str], labels: Optional[List[str]] = None, out_path: str = 'failure_rates.png') -> None:
	"""为每个路径绘制累计失败率折线图和95%置信区间的阴影半宽度。

	paths: 列表，每项为文件或目录（参见 `_load_sequence` 的加载规则）。
	labels: 可选的曲线标签列表。
	out_path: 输出图片路径。
	"""
	z = norm.ppf(1 - 0.05 / 2)  # 95% CI

	plt.figure(figsize=(10, 6), dpi=120)

	if labels is None:
		labels = [os.path.basename(p) or p for p in paths]

	colors = plt.get_cmap('tab10')
	# 红色与 tests/draw_RHF.py 中一致
	primary_red = '#8B0000'
	primary_fill = (1.0, 215/255, 215/255)

	# 按用户要求：横坐标为 `paths` 的索引（iteration），每个路径聚合为一个点（合并目录下所有 .npy）
	means = []
	lowers = []
	uppers = []
	xs = list(range(len(paths)))

	for idx, path in enumerate(paths):
		data = _load_sequence(path)

		arr = np.asarray(data).ravel()
		if arr.size == 0:
			print(f"Warning: empty data for {path}, skipped.")
			means.append(np.nan)
			lowers.append(np.nan)
			uppers.append(np.nan)
			continue

		N = arr.size
		mean = float(np.nanmean(arr))
		# 标准误：对二值数据或加权数据都使用样本标准差 / sqrt(N)
		if N > 1:
			se = float(np.nanstd(arr, ddof=1) / np.sqrt(N))
		else:
			se = 0.0
		half = float(z * se)

		lower = mean - half
		upper = mean + half
		# 对概率值做裁剪（若数据为概率/失败率）
		lower = max(lower, 0.0)
		upper = min(upper, 1.0)

		means.append(mean)
		lowers.append(lower)
		uppers.append(upper)

	means = np.array(means)
	lowers = np.array(lowers)
	uppers = np.array(uppers)

	# 绘制线与置信区间带
	for idx in range(len(paths)):
		if idx == 0:
			line_color = primary_red
			fill_color = primary_fill
			alpha = 0.5
		else:
			line_color = colors(idx % 10)
			fill_color = line_color
			alpha = 0.25

	plt.plot(xs, means, '-o', color=primary_red, linewidth=2)
	#plt.fill_between(xs, lowers, uppers, color=primary_fill, alpha=0.5)
	# 标签显示每个点的值
	for xi, m in zip(xs, means):
		if not np.isnan(m):
			plt.text(xi, m, f"{m:.2e}", ha='center', va='bottom', fontsize=14)

	plt.xlabel('Iteration', fontsize=16)
	plt.ylabel('Failure rate', fontsize=16)
	plt.legend(fontsize=14)
	plt.grid(True, alpha=0.3)
	plt.ylim(1e-2, 8e-2)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)

	out_dir = os.path.dirname(out_path)
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok=True)

	plt.tight_layout()
	plt.savefig(out_path, dpi=300)
	plt.close('all')


if __name__ == '__main__':
	# 将结果列表作为输入，输出到 workspace 下的 results/failure_rates.png
	results_list = ['all_results/results_origin', 'all_results/results_round1', 'all_results/results_round2', 'all_results/results_round3', 'all_results/results_round4', 'all_results/results_round5']
	plot_failure_rates(results_list, out_path='training/failure_rates.png')
	print('Saved failure plot to training/failure_rates.png')