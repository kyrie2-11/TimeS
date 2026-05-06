"""
隶属度映射模块
=============
实现从距离到隶属度的映射，支持多通道加权距离与自适应尺度。
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import euclidean


class MembershipMapper:
    """隶属度映射器。"""

    def __init__(
        self,
        shapelets: List[List[Dict]],
        shapelets_info: np.ndarray,
        beta_mode: str = 'adaptive',
        global_beta: float = 2.0,
        channel_weights: Optional[np.ndarray] = None,
        use_multichannel: bool = True,
    ):
        self.shapelets = shapelets
        self.shapelets_info = shapelets_info
        self.beta_mode = beta_mode
        self.global_beta = global_beta
        self.channel_weights = channel_weights
        self.use_multichannel = use_multichannel

        self.n_classes = len(shapelets)
        self.n_per_class = len(shapelets[0]) if shapelets else 0

        self.adaptive_betas = {}

        print('[MembershipMapper] 初始化完成')
        print(f'  - β 模式: {beta_mode}')
        print(f'  - 多通道距离: {use_multichannel}')
        print(f'  - Shapelet 数: {self.n_classes} 类 × {self.n_per_class} 个/类')

    def learn_channel_weights(self, train_data: np.ndarray, mode: str = 'inverse_mad', eps: float = 1e-8):
        """基于训练正常样本学习通道权重。"""
        _, n_dims, _ = train_data.shape

        if mode == 'uniform' or n_dims == 1:
            self.channel_weights = np.ones(n_dims, dtype=float) / n_dims
            return

        # 稳健统计：MAD 越小表示该通道正常形态更稳定，权重更高
        med = np.median(train_data, axis=(0, 2))
        mad = np.median(np.abs(train_data - med[None, :, None]), axis=(0, 2))
        weights = 1.0 / (mad + eps)
        weights = np.maximum(weights, eps)
        self.channel_weights = weights / np.sum(weights)

        print('[MembershipMapper] 已学习通道权重')
        print(f'  - 权重: {np.round(self.channel_weights, 4)}')

    def calibrate_adaptive_beta(self, train_data: np.ndarray, eps: float = 1e-8):
        """为每个 shapelet 校准自适应 β。"""
        if self.beta_mode != 'adaptive':
            print('[MembershipMapper] 非自适应模式，跳过校准')
            return

        print('[MembershipMapper] 校准自适应 β...')

        n_samples = train_data.shape[0]

        for class_idx in range(self.n_classes):
            for shapelet_idx in range(self.n_per_class):
                shapelet = self.shapelets[class_idx][shapelet_idx]
                dim = int(self.shapelets_info[class_idx, shapelet_idx, 5])

                sample_min_dists = []
                for sample_idx in range(n_samples):
                    ts = train_data[sample_idx]
                    distances, _ = self._sliding_distance(ts, shapelet, dim)
                    if len(distances) > 0:
                        sample_min_dists.append(np.min(distances))

                if len(sample_min_dists) > 0:
                    m_s = np.median(sample_min_dists)
                    beta_s = np.log(2) / (m_s + eps)
                else:
                    beta_s = self.global_beta

                self.adaptive_betas[(class_idx, shapelet_idx)] = beta_s

        print('[MembershipMapper] ✓ 校准完成')
        if len(self.adaptive_betas) > 0:
            print(f"  - 平均 β: {np.mean(list(self.adaptive_betas.values())):.4f}")

    def _ensure_seq_first(self, time_series: np.ndarray) -> np.ndarray:
        """统一到 (seq_len, n_dims)。"""
        if time_series.ndim != 2:
            raise ValueError('time_series 必须是二维数组')

        # 约定输入常见为 (n_dims, seq_len)
        if time_series.shape[0] < time_series.shape[1]:
            return time_series.T
        return time_series

    def _extract_shapelet_components(self, shapelet, fallback_dim: int):
        """兼容旧版 1D shapelet 与新版 dict shapelet。"""
        if isinstance(shapelet, dict):
            pattern = shapelet.get('pattern')
            context = shapelet.get('context')
            dim = int(shapelet.get('dim', fallback_dim))
        else:
            pattern = shapelet
            context = None
            dim = fallback_dim

        if pattern is None:
            raise ValueError('shapelet pattern 为空')

        pattern = np.asarray(pattern, dtype=float)
        return pattern, context, dim

    def _sliding_distance(
        self,
        time_series: np.ndarray,
        shapelet,
        dim: int,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """滑窗计算距离，支持多通道加权。"""
        ts = self._ensure_seq_first(time_series)
        seq_len, n_dims = ts.shape

        pattern, context, main_dim = self._extract_shapelet_components(shapelet, dim)
        shapelet_len = len(pattern)

        if shapelet_len > seq_len:
            return np.array([np.inf]), np.array([0])

        T = seq_len - shapelet_len + 1
        distances = np.zeros(T)

        if self.channel_weights is None or len(self.channel_weights) != n_dims:
            channel_weights = np.ones(n_dims, dtype=float) / n_dims
        else:
            channel_weights = self.channel_weights

        pattern_norm = self._z_normalize(pattern)

        for t in range(T):
            window = ts[t:t + shapelet_len, :]

            if self.use_multichannel and context is not None and context.shape[0] == n_dims:
                # 多通道加权距离
                per_dim_sq = 0.0
                for d in range(n_dims):
                    subseq = self._z_normalize(window[:, d])
                    ref = self._z_normalize(context[d])
                    dist_d = euclidean(subseq, ref)
                    per_dim_sq += channel_weights[d] * (dist_d ** 2)
                dist = np.sqrt(per_dim_sq)
            else:
                # 退化为主通道距离
                subseq = self._z_normalize(window[:, main_dim])
                dist = euclidean(subseq, pattern_norm)

            if normalize:
                dist = dist / np.sqrt(shapelet_len)

            distances[t] = dist

        positions = np.arange(T)
        return distances, positions

    @staticmethod
    def _z_normalize(sequence: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mean = np.mean(sequence)
        std = np.std(sequence)
        if std < eps:
            return sequence - mean
        return (sequence - mean) / std

    def compute_memberships(self, time_series: np.ndarray) -> Dict:
        """计算与所有 shapelet 的隶属度。"""
        memberships_dict = {}

        for class_idx in range(self.n_classes):
            class_memberships = {}

            for shapelet_idx in range(self.n_per_class):
                if shapelet_idx >= len(self.shapelets[class_idx]):
                    continue

                shapelet = self.shapelets[class_idx][shapelet_idx]
                dim = int(self.shapelets_info[class_idx, shapelet_idx, 5])

                distances, positions = self._sliding_distance(time_series, shapelet, dim)

                if self.beta_mode == 'adaptive':
                    beta = self.adaptive_betas.get((class_idx, shapelet_idx), self.global_beta)
                else:
                    beta = self.global_beta

                memberships = np.exp(-beta * distances)
                max_idx = int(np.argmax(memberships))

                class_memberships[f'shapelet_{shapelet_idx}'] = {
                    'memberships': memberships,
                    'positions': positions,
                    'distances': distances,
                    'beta': beta,
                    'max_membership': float(memberships[max_idx]),
                    'max_pos': int(positions[max_idx]),
                    'dim': dim,
                    'info_gain': float(self.shapelets_info[class_idx, shapelet_idx, 3]),
                }

            memberships_dict[f'class_{class_idx}'] = class_memberships

        return memberships_dict

    def batch_compute_memberships(self, time_series_batch: np.ndarray) -> List[Dict]:
        return [self.compute_memberships(ts) for ts in time_series_batch]


def compute_memberships(
    time_series: np.ndarray,
    shapelets: List[List[Dict]],
    shapelets_info: np.ndarray,
    beta_mode: str = 'adaptive',
    train_data: Optional[np.ndarray] = None,
    use_multichannel: bool = True,
) -> Dict:
    """便捷函数：计算隶属度。"""
    mapper = MembershipMapper(
        shapelets,
        shapelets_info,
        beta_mode=beta_mode,
        use_multichannel=use_multichannel,
    )

    if train_data is not None:
        mapper.learn_channel_weights(train_data)

    if beta_mode == 'adaptive' and train_data is not None:
        mapper.calibrate_adaptive_beta(train_data)

    return mapper.compute_memberships(time_series)


if __name__ == '__main__':
    print('隶属度映射模块测试')
    print('=' * 50)

    np.random.seed(42)
    n_classes = 1
    n_per_class = 2
    l = 20
    n_dims = 3

    shapelets = [[{
        'pattern': np.random.randn(l),
        'context': np.random.randn(n_dims, l),
        'dim': 0,
        'length': l,
    } for _ in range(n_per_class)] for _ in range(n_classes)]

    shapelets_info = np.zeros((n_classes, n_per_class, 6), dtype=float)
    test_ts = np.random.randn(n_dims, 100)

    memberships = compute_memberships(test_ts, shapelets, shapelets_info, beta_mode='global')
    print(f"类别数: {len(memberships)}")
