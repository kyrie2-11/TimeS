"""
Shapelet 池构建模块
================
支持两种构建模式：
1. 监督模式：沿用 OSD 信息增益 shapelet
2. One-class 模式：基于正常样本稳定性与代表性筛选 shapelet
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os

# 导入项目中的 shapelet 发现模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Shapelet.mul_shapelet_discovery import ShapeletDiscover


class ShapeletPool:
    """Shapelet 池管理器"""

    def __init__(
        self,
        window_size: int = 30,
        num_pip: float = 0.2,
        n_per_class: int = 3,
        processes: int = 4,
        random_state: int = 42,
    ):
        self.window_size = window_size
        self.num_pip = num_pip
        self.n_per_class = n_per_class
        self.processes = processes
        self.random_state = random_state

        self.shapelets = None
        self.shapelets_info = None
        self.n_classes = None

    def build(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        p: float = 0.1,
        pi: float = 0.1,
        one_class_mode: str = 'auto',
        max_candidates: int = 400,
        diversity_threshold: float = 0.9,
    ) -> Tuple[List[List[Dict]], np.ndarray]:
        """
        构建 shapelet 池

        one_class_mode:
            - 'auto': 标签只有 1 类时启用 one-class 筛选
            - 'force': 强制启用 one-class 筛选
            - 'off': 关闭 one-class 筛选，使用监督 OSD
        """
        unique_labels = np.unique(train_labels)
        self.n_classes = len(unique_labels)

        print("[ShapeletPool] 开始构建 shapelet 池")
        print(f"  - 训练样本数: {train_data.shape[0]}")
        print(f"  - 维度数: {train_data.shape[1]}")
        print(f"  - 序列长度: {train_data.shape[2]}")
        print(f"  - 类别数: {self.n_classes}")
        print(f"  - 每类 shapelet 数: {self.n_per_class}")

        use_one_class = (
            one_class_mode == 'force' or
            (one_class_mode == 'auto' and self.n_classes == 1)
        )

        if use_one_class:
            print("[ShapeletPool] 使用 one-class 稳定性筛选模式")
            shapelets, shapelets_info = self._build_one_class_pool(
                train_data,
                train_labels,
                max_candidates=max_candidates,
                diversity_threshold=diversity_threshold,
            )
        else:
            print("[ShapeletPool] 使用 OSD 监督筛选模式")
            shapelets, shapelets_info = self._build_supervised_pool(
                train_data,
                train_labels,
                p=p,
                pi=pi,
            )

        self.shapelets = shapelets
        self.shapelets_info = shapelets_info

        print("[ShapeletPool] ✓ Shapelet 池构建完成")
        print(f"  - 总 shapelet 数: {sum(len(s) for s in self.shapelets)}")
        return self.shapelets, self.shapelets_info

    def _build_supervised_pool(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        p: float,
        pi: float,
    ) -> Tuple[List[List[Dict]], np.ndarray]:
        """监督模式：使用 OSD 信息增益。"""
        n_samples, n_dims, seq_len = train_data.shape

        shapelet_discovery = ShapeletDiscover(
            window_size=self.window_size,
            num_pip=self.num_pip,
            processes=self.processes,
            len_of_ts=seq_len,
            dim=n_dims,
        )

        print("[ShapeletPool] 提取候选片段...")
        shapelet_discovery.extract_candidate(train_data)

        print("[ShapeletPool] 评估候选片段（信息增益）...")
        shapelet_discovery.discovery(train_data, train_labels)

        shapelets_info_raw = shapelet_discovery.get_shapelet_info(
            number_of_shapelet=self.n_per_class,
            p=p,
            pi=pi,
        )

        unique_labels = np.unique(train_labels)
        shapelets = []
        shapelets_info_3d = []

        for class_label in unique_labels:
            class_shapelets = []
            class_info = shapelets_info_raw[shapelets_info_raw[:, 4] == class_label]

            if len(class_info) == 0:
                dummy_info = np.zeros((self.n_per_class, 6), dtype=float)
                for i in range(self.n_per_class):
                    ts_pos = i % n_samples
                    start = 0
                    end = min(self.window_size, seq_len)
                    dim_idx = 0
                    sh = self._build_shapelet_dict(train_data, ts_pos, start, end, dim_idx)
                    class_shapelets.append(sh)
                    dummy_info[i] = [ts_pos, start, end, 0.0, class_label, dim_idx]
                shapelets_info_3d.append(dummy_info)
            else:
                n_available = min(len(class_info), self.n_per_class)
                class_info_padded = np.zeros((self.n_per_class, 6), dtype=float)

                for i in range(self.n_per_class):
                    info = class_info[i] if i < n_available else class_info[n_available - 1]
                    ts_pos = int(info[0])
                    start = int(info[1])
                    end = int(info[2])
                    dim_idx = int(info[5])

                    sh = self._build_shapelet_dict(train_data, ts_pos, start, end, dim_idx)
                    class_shapelets.append(sh)
                    class_info_padded[i] = info

                shapelets_info_3d.append(class_info_padded)

            shapelets.append(class_shapelets)

        return shapelets, np.array(shapelets_info_3d, dtype=float)

    def _build_one_class_pool(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        max_candidates: int,
        diversity_threshold: float,
    ) -> Tuple[List[List[Dict]], np.ndarray]:
        """
        one-class 模式：用正常样本的代表性与稳定性选 shapelet。

        候选打分：
            score = -mean(min_dist) - 0.5 * std(min_dist)
        分数越大越好（越接近正常模板且跨样本稳定）。
        """
        n_samples, n_dims, seq_len = train_data.shape
        rng = np.random.default_rng(self.random_state)

        candidate_pool = []
        max_start = max(1, seq_len - self.window_size + 1)

        # 均匀随机采样候选，避免全量组合导致开销过大
        for _ in range(max_candidates):
            ts_pos = int(rng.integers(0, n_samples))
            dim_idx = int(rng.integers(0, n_dims))
            start = int(rng.integers(0, max_start))
            end = start + self.window_size

            pattern = self._z_normalize(train_data[ts_pos, dim_idx, start:end])
            mean_min_dist, std_min_dist = self._candidate_normal_distance_stats(
                train_data,
                pattern,
                dim_idx,
            )

            score = -(mean_min_dist + 0.5 * std_min_dist)
            candidate_pool.append(
                {
                    'ts_pos': ts_pos,
                    'start': start,
                    'end': end,
                    'dim': dim_idx,
                    'score': score,
                    'pattern': pattern,
                }
            )

        candidate_pool = sorted(candidate_pool, key=lambda x: x['score'], reverse=True)

        selected = []
        for cand in candidate_pool:
            if len(selected) >= self.n_per_class:
                break

            if self._is_too_similar(cand['pattern'], selected, diversity_threshold):
                continue
            selected.append(cand)

        # 多样性过滤后不足则补齐
        idx = 0
        while len(selected) < self.n_per_class and idx < len(candidate_pool):
            selected.append(candidate_pool[idx])
            idx += 1

        class_shapelets = []
        class_info = np.zeros((self.n_per_class, 6), dtype=float)
        class_label = float(np.unique(train_labels)[0])

        for i, cand in enumerate(selected[:self.n_per_class]):
            ts_pos = int(cand['ts_pos'])
            start = int(cand['start'])
            end = int(cand['end'])
            dim_idx = int(cand['dim'])
            score = float(cand['score'])

            sh = self._build_shapelet_dict(train_data, ts_pos, start, end, dim_idx)
            class_shapelets.append(sh)
            class_info[i] = [ts_pos, start, end, score, class_label, dim_idx]

        return [class_shapelets], np.array([class_info], dtype=float)

    def _candidate_normal_distance_stats(
        self,
        train_data: np.ndarray,
        pattern: np.ndarray,
        dim_idx: int,
    ) -> Tuple[float, float]:
        """统计候选 shapelet 在正常样本上的最小距离均值与方差。"""
        min_dists = []
        for i in range(train_data.shape[0]):
            series = train_data[i, dim_idx]
            min_dists.append(self._min_sliding_distance_1d(series, pattern))
        min_dists = np.array(min_dists, dtype=float)
        return float(np.mean(min_dists)), float(np.std(min_dists))

    def _min_sliding_distance_1d(self, series: np.ndarray, pattern: np.ndarray) -> float:
        """1D 滑窗最小距离。"""
        l = len(pattern)
        if len(series) < l:
            return float(np.inf)

        best = np.inf
        for t in range(len(series) - l + 1):
            subseq = self._z_normalize(series[t:t + l])
            dist = np.sqrt(np.mean((subseq - pattern) ** 2))
            if dist < best:
                best = dist
        return float(best)

    def _is_too_similar(self, pattern: np.ndarray, selected: List[Dict], threshold: float) -> bool:
        """避免 shapelet 池中出现高度冗余片段。"""
        for item in selected:
            p = item['pattern']
            if len(p) != len(pattern):
                continue
            corr = np.corrcoef(pattern, p)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            if abs(corr) >= threshold:
                return True
        return False

    def _build_shapelet_dict(
        self,
        train_data: np.ndarray,
        ts_pos: int,
        start: int,
        end: int,
        dim_idx: int,
    ) -> Dict:
        """构建包含主通道与多通道上下文的 shapelet。"""
        pattern = self._z_normalize(train_data[ts_pos, dim_idx, start:end])

        context = train_data[ts_pos, :, start:end].copy()
        for d in range(context.shape[0]):
            context[d] = self._z_normalize(context[d])

        return {
            'pattern': pattern,
            'context': context,
            'dim': int(dim_idx),
            'length': int(end - start),
        }

    @staticmethod
    def _z_normalize(sequence: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mean = np.mean(sequence)
        std = np.std(sequence)
        if std < eps:
            return sequence - mean
        return (sequence - mean) / std

    def get_pool_summary(self) -> dict:
        if self.shapelets is None:
            return {'status': '未构建'}

        lengths = []
        info_scores = []

        for class_idx in range(len(self.shapelets)):
            for shapelet_idx in range(len(self.shapelets[class_idx])):
                shapelet = self.shapelets[class_idx][shapelet_idx]
                lengths.append(shapelet.get('length', len(shapelet.get('pattern', []))))
                info_scores.append(self.shapelets_info[class_idx, shapelet_idx, 3])

        return {
            'n_classes': len(self.shapelets),
            'n_per_class': self.n_per_class,
            'total_shapelets': len(lengths),
            'avg_length': np.mean(lengths) if lengths else 0,
            'min_length': np.min(lengths) if lengths else 0,
            'max_length': np.max(lengths) if lengths else 0,
            'avg_score': np.mean(info_scores) if info_scores else 0,
            'min_score': np.min(info_scores) if info_scores else 0,
            'max_score': np.max(info_scores) if info_scores else 0,
        }


def build_shapelet_pool(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    n_per_class: int = 3,
    window_size: int = 30,
    num_pip: float = 0.2,
    one_class_mode: str = 'auto',
    random_state: int = 42,
) -> Tuple[List[List[Dict]], np.ndarray]:
    """便捷函数：构建 shapelet 池。"""
    pool = ShapeletPool(
        window_size=window_size,
        num_pip=num_pip,
        n_per_class=n_per_class,
        random_state=random_state,
    )

    return pool.build(
        train_data,
        train_labels,
        one_class_mode=one_class_mode,
    )


if __name__ == '__main__':
    print('Shapelet 池构建模块测试')
    print('=' * 50)

    np.random.seed(42)
    n_samples = 16
    n_dims = 3
    seq_len = 120

    train_data = np.random.randn(n_samples, n_dims, seq_len)
    train_labels = np.zeros(n_samples, dtype=int)

    shapelets, shapelets_info = build_shapelet_pool(
        train_data,
        train_labels,
        n_per_class=4,
        window_size=24,
        one_class_mode='auto',
    )

    print('构建完成')
    print(f"shapelet 结构: {len(shapelets)} 类 x {len(shapelets[0])} 个")
    print(f"info shape: {shapelets_info.shape}")
