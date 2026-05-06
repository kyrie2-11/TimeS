"""
改进的多维Shapelet发现模块
支持：
1. 维度权重学习（基于MAD稳定性和相关性）
2. 多维联合采样（单维 + 多维组合）
3. 多维联合距离度量
4. 多尺度Shapelet提取
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations
from typing import List, Tuple, Dict


class MultivariateShapeletDiscover:
    """
    多维Shapelet发现器
    支持跨维度的联合Shapelet提取
    """
    
    def __init__(self, window_size: int = 30, num_pip: float = 0.2,
                 univariate_ratio: float = 0.7, multivariate_ratio: float = 0.3,
                 max_multivariate_dims: int = 3):
        """
        初始化多维Shapelet发现器
        
        Args:
            window_size: Shapelet窗口大小
            num_pip: 采样比例
            univariate_ratio: 单维Shapelet比例 (默认70%)
            multivariate_ratio: 多维Shapelet比例 (默认30%)
            max_multivariate_dims: 多维Shapelet的最大维度组合数 (默认3)
        """
        self.window_size = window_size
        self.num_pip = num_pip
        self.univariate_ratio = univariate_ratio
        self.multivariate_ratio = multivariate_ratio
        self.max_multivariate_dims = max_multivariate_dims
        
        self.candidates = []
        self.candidates_info = []
        self.channel_importance = None
        self.channel_weights = None
        
    def compute_channel_importance(self, train_data: np.ndarray) -> np.ndarray:
        """
        计算维度重要性权重（基于MAD和相关性）
        
        Args:
            train_data: [n_samples, n_dims, seq_len] 训练数据
        
        Returns:
            weights: [n_dims] 维度权重（和为1）
        """
        n_samples, n_dims, seq_len = train_data.shape
        
        # 1. 计算MAD (Median Absolute Deviation)
        mad_scores = np.zeros(n_dims)
        for d in range(n_dims):
            data_d = train_data[:, d, :].flatten()
            median = np.median(data_d)
            mad = np.median(np.abs(data_d - median))
            mad_scores[d] = 1.0 / (mad + 1e-6)  # 倒数关系：MAD越小，权重越高
        
        # 2. 计算相关性（基于维度间的均值相似度）
        corr_scores = np.zeros(n_dims)
        means = np.mean(train_data, axis=(0, 2))  # [n_dims]
        mean_std = np.std(means)
        
        for d in range(n_dims):
            # 与其他维度的相关性（用标准差表示，小则权重高）
            diff_from_mean = abs(means[d] - np.mean(means))
            corr_scores[d] = 1.0 / (diff_from_mean + mean_std + 1e-6)
        
        # 3. 归一化并组合
        mad_normalized = mad_scores / np.sum(mad_scores)
        corr_normalized = corr_scores / np.sum(corr_scores)
        
        # 加权组合：70% MAD, 30% 相关性
        weights = 0.7 * mad_normalized + 0.3 * corr_normalized
        weights = weights / np.sum(weights)  # 重新归一化
        
        self.channel_importance = weights
        self.channel_weights = weights
        
        print(f"\n维度重要性权重（window_size={self.window_size}）:")
        for d in range(min(n_dims, 10)):
            print(f"  维度{d:2d}: {weights[d]:.4f}")
        if n_dims > 10:
            print(f"  ... 还有 {n_dims-10} 个维度")
        
        return weights
    
    def extract_candidate(self, train_data: np.ndarray):
        """
        提取候选Shapelet（支持单维和多维）
        
        Args:
            train_data: [n_samples, n_dims, seq_len]
        """
        n_samples, n_dims, seq_len = train_data.shape
        
        if self.channel_importance is None:
            self.compute_channel_importance(train_data)
        
        n_candidates_per_sample = max(1, int(seq_len * self.num_pip))
        max_total_candidates = 100
        samples_to_use = min(n_samples, max(5, max_total_candidates // (n_dims * n_candidates_per_sample)))
        
        self.candidates = []
        self.candidates_info = []
        
        print(f"\n多维Shapelet采样（window_size={self.window_size}):")
        print(f"  单维Shapelet比例: {self.univariate_ratio*100:.0f}%")
        print(f"  多维Shapelet比例: {self.multivariate_ratio*100:.0f}%")
        print(f"  从{n_samples}个样本中选择{samples_to_use}个")
        
        # 随机选择样本
        sample_indices = np.random.choice(n_samples, size=samples_to_use, replace=False)
        
        # 计算目标候选数
        target_univariate = int(max_total_candidates * self.univariate_ratio)
        target_multivariate = max_total_candidates - target_univariate
        
        # ========== 单维Shapelet采样 ==========
        univariate_count = 0
        while univariate_count < target_univariate:
            sample_idx = np.random.choice(sample_indices)
            dim_idx = np.random.choice(n_dims, p=self.channel_weights)
            
            max_start = seq_len - self.window_size
            if max_start <= 0:
                continue
            
            start = np.random.randint(0, max_start)
            end = start + self.window_size
            
            shapelet = train_data[sample_idx, dim_idx, start:end].copy()
            shapelet = self._z_normalize_shapelet(shapelet)
            
            self.candidates.append(shapelet)
            self.candidates_info.append({
                'sample_idx': sample_idx,
                'start': start,
                'end': end,
                'dims': [dim_idx],  # 单维：列表形式
                'info_gain': 0.0,
                'label': 0,
                'is_multivariate': False
            })
            univariate_count += 1
        
        # ========== 多维Shapelet采样 ==========
        multivariate_count = 0
        max_attempts = target_multivariate * 5  # 防止无限循环
        attempts = 0
        
        while multivariate_count < target_multivariate and attempts < max_attempts:
            attempts += 1
            
            # 随机选择2-3个维度（基于权重）
            k = np.random.randint(2, min(self.max_multivariate_dims + 1, n_dims + 1))
            selected_dims = np.random.choice(n_dims, size=k, replace=False, p=self.channel_weights).tolist()
            selected_dims.sort()  # 排序保证一致性
            
            sample_idx = np.random.choice(sample_indices)
            max_start = seq_len - self.window_size
            if max_start <= 0:
                continue
            
            start = np.random.randint(0, max_start)
            end = start + self.window_size
            
            # 提取多维Shapelet并拼接
            shapelet_parts = []
            for dim_idx in selected_dims:
                part = train_data[sample_idx, dim_idx, start:end].copy()
                shapelet_parts.append(self._z_normalize_shapelet(part))
            
            shapelet = np.concatenate(shapelet_parts)  # [window_size * k]
            
            self.candidates.append(shapelet)
            self.candidates_info.append({
                'sample_idx': sample_idx,
                'start': start,
                'end': end,
                'dims': selected_dims,  # 多维：维度列表
                'info_gain': 0.0,
                'label': 0,
                'is_multivariate': True
            })
            multivariate_count += 1
        
        print(f"  ✓ 提取了 {len(self.candidates)} 个候选Shapelet")
        print(f"    - 单维: {univariate_count}")
        print(f"    - 多维: {multivariate_count}")
    
    def _z_normalize_shapelet(self, shapelet: np.ndarray) -> np.ndarray:
        """Z-normalize单个Shapelet"""
        mean = np.mean(shapelet)
        std = np.std(shapelet)
        if std > 1e-8:
            return (shapelet - mean) / std
        return shapelet
    
    def multivariate_distance(self, ts_multiview: np.ndarray, shapelet: np.ndarray,
                            dims: List[int], weights: np.ndarray = None) -> float:
        """
        计算多维时间序列与Shapelet的最小滑窗距离
        （优化版：支持跳跃采样加速）
        
        Args:
            ts_multiview: [n_dims, seq_len] 多维时间序列
            shapelet: 标准化后的Shapelet（单维或多维拼接）
            dims: Shapelet涉及的维度索引列表
            weights: [n_dims] 通道权重（暂未使用）
        
        Returns:
            min_dist: 最小距离
        """
        if weights is None:
            weights = np.ones(ts_multiview.shape[0]) / ts_multiview.shape[0]
        
        n_dims_in_shapelet = len(dims)
        window_size = len(shapelet) // n_dims_in_shapelet
        seq_len = ts_multiview.shape[1]
        
        if window_size > seq_len:
            return float('inf')
        
        # 优化：跳跃采样加速（对于长序列）
        n_positions = seq_len - window_size + 1
        step = max(1, n_positions // 50)  # 最多采样50个位置
        
        min_dist = float('inf')
        
        # 滑窗计算距离（优化版）
        for t in range(0, n_positions, step):
            # 提取窗口（多维）
            window_parts = []
            for d_idx in dims:
                window = ts_multiview[d_idx, t:t + window_size]
                window = self._z_normalize_shapelet(window)
                window_parts.append(window)
            
            window_concat = np.concatenate(window_parts)
            
            # 计算欧氏距离
            dist = np.sqrt(np.sum((window_concat - shapelet) ** 2))
            
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def discovery(self, train_data: np.ndarray, train_labels: np.ndarray):
        """
        使用信息增益评估候选Shapelet
        
        Args:
            train_data: [n_samples, n_dims, seq_len]
            train_labels: [n_samples]
        """
        n_samples = len(train_data)
        n_candidates = len(self.candidates)
        n_dims = train_data.shape[1]
        
        if n_candidates == 0:
            print("警告：没有候选Shapelet")
            return
        
        # 计算距离矩阵
        print(f"\n计算距离矩阵（{n_candidates} x {n_samples})...")
        distance_matrix = np.zeros((n_candidates, n_samples))
        
        for cand_idx in range(n_candidates):
            if cand_idx % max(1, n_candidates//10) == 0:
                print(f"  进度: {cand_idx}/{n_candidates} ({100*cand_idx//n_candidates}%)")
            
            candidate_info = self.candidates_info[cand_idx]
            shapelet = self.candidates[cand_idx]
            dims = candidate_info['dims']
            
            for sample_idx in range(n_samples):
                ts = train_data[sample_idx]  # [n_dims, seq_len]
                dist = self.multivariate_distance(ts, shapelet, dims, self.channel_weights)
                distance_matrix[cand_idx, sample_idx] = dist
        
        # 信息增益评估
        print(f"计算信息增益...")
        for cand_idx in range(n_candidates):
            X = distance_matrix[cand_idx].reshape(-1, 1)
            dt = DecisionTreeClassifier(max_depth=1, random_state=42)
            dt.fit(X, train_labels)
            
            score = dt.score(X, train_labels)
            self.candidates_info[cand_idx]['info_gain'] = score
            
            predicted_labels = dt.predict(X)
            most_common_label = np.bincount(predicted_labels.astype(int)).argmax() if len(np.bincount(predicted_labels.astype(int))) > 0 else 0
            self.candidates_info[cand_idx]['label'] = most_common_label
        
        print(f"✓ 信息增益计算完成")
    
    def get_shapelets_by_class(self, n_per_class: int = 3) -> Dict:
        """
        按类别返回top-k的Shapelet
        
        Returns:
            dict: {class_label: [shapelet_indices]}
        """
        shapelets_by_class = {}
        
        for cand_idx, info in enumerate(self.candidates_info):
            label = int(info['label'])
            if label not in shapelets_by_class:
                shapelets_by_class[label] = []
            
            shapelets_by_class[label].append((cand_idx, info['info_gain']))
        
        # 按信息增益排序并取top-n
        result = {}
        for label in shapelets_by_class:
            sorted_candidates = sorted(shapelets_by_class[label], key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in sorted_candidates[:n_per_class]]
            result[label] = top_indices
        
        print(f"\n按类别选择top-{n_per_class}的Shapelet:")
        for label in sorted(result.keys()):
            count_univariate = sum(1 for idx in result[label] if not self.candidates_info[idx]['is_multivariate'])
            count_multivariate = len(result[label]) - count_univariate
            print(f"  类别{label}: {len(result[label])}个Shapelet (单维:{count_univariate}, 多维:{count_multivariate})")
        
        return result
    
    def get_selected_shapelets(self, indices: List[int]) -> List[Dict]:
        """
        返回选定的Shapelet及其元信息
        """
        result = []
        for idx in indices:
            result.append({
                'shapelet': self.candidates[idx],
                'dims': self.candidates_info[idx]['dims'],
                'info_gain': self.candidates_info[idx]['info_gain'],
                'label': self.candidates_info[idx]['label'],
                'is_multivariate': self.candidates_info[idx]['is_multivariate']
            })
        return result
