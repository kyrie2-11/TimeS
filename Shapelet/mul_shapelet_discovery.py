"""
简化的 Shapelet 发现模块
使用随机采样 + 信息增益的方法提取 shapelet
适用于快速原型和实验
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import List, Tuple


class ShapeletDiscover:
    """
    简化的 Shapelet 发现器
    使用随机采样 + 决策树信息增益选择最佳 shapelet
    """
    
    def __init__(self, window_size: int = 30, num_pip: float = 0.2,
                 processes: int = 4, len_of_ts: int = 100, dim: int = 1):
        """
        初始化
        
        Args:
            window_size: shapelet 窗口大小
            num_pip: 采样比例
            processes: 并行进程数（暂未使用）
            len_of_ts: 时间序列长度
            dim: 维度数
        """
        self.window_size = window_size
        self.num_pip = num_pip
        self.processes = processes
        self.len_of_ts = len_of_ts
        self.dim = dim
        self.candidates = []
        self.candidates_info = []
        
    def extract_candidate(self, train_data: np.ndarray):
        """
        提取候选 shapelet
        
        Args:
            train_data: (n_samples, n_dims, seq_len)
        """
        n_samples, n_dims, seq_len = train_data.shape
        n_candidates_per_sample = max(1, int(seq_len * self.num_pip))
        
        # 限制总候选数量以加快速度
        max_total_candidates = 50  # 大幅减少候选数量
        samples_to_use = min(n_samples, max(5, max_total_candidates // (n_dims * n_candidates_per_sample)))
        
        self.candidates = []
        self.candidates_info = []
        
        print(f"  采样策略: 从 {n_samples} 个样本中选择 {samples_to_use} 个")
        
        # 随机选择样本索引
        sample_indices = np.random.choice(n_samples, size=samples_to_use, replace=False)
        
        for sample_idx in sample_indices:
            for dim_idx in range(n_dims):
                # 随机采样起始位置
                max_start = seq_len - self.window_size
                if max_start <= 0:
                    continue
                
                start_positions = np.random.choice(
                    max_start, 
                    size=min(n_candidates_per_sample, max_start),
                    replace=False
                )
                
                for start in start_positions:
                    end = start + self.window_size
                    shapelet = train_data[sample_idx, dim_idx, start:end].copy()
                    
                    # Z-normalize
                    shapelet_mean = np.mean(shapelet)
                    shapelet_std = np.std(shapelet)
                    if shapelet_std > 1e-8:
                        shapelet = (shapelet - shapelet_mean) / shapelet_std
                    
                    self.candidates.append(shapelet)
                    self.candidates_info.append([
                        sample_idx,  # ts_pos
                        start,       # start
                        end,         # end
                        0.0,         # info_gain (待计算)
                        0,           # label (待填充)
                        dim_idx      # dim
                    ])
        
        self.candidates = np.array(self.candidates)
        self.candidates_info = np.array(self.candidates_info)
        
        print(f"  提取了 {len(self.candidates)} 个候选 shapelet")
    
    def discovery(self, train_data: np.ndarray, train_labels: np.ndarray):
        """
        使用信息增益评估候选 shapelet
        
        Args:
            train_data: (n_samples, n_dims, seq_len)
            train_labels: (n_samples,)
        """
        n_samples = len(train_data)
        n_candidates = len(self.candidates)
        
        if n_candidates == 0:
            print("  警告: 没有候选 shapelet")
            return
        
        # 计算每个候选 shapelet 与所有训练样本的距离
        print(f"  计算距离矩阵 ({n_candidates} x {n_samples})...")
        distance_matrix = np.zeros((n_candidates, n_samples))
        
        for cand_idx in range(n_candidates):
            if cand_idx % 10 == 0:
                print(f"    进度: {cand_idx}/{n_candidates} ({100*cand_idx//n_candidates}%)")
            
            shapelet = self.candidates[cand_idx]
            dim_idx = int(self.candidates_info[cand_idx, 5])
            
            for sample_idx in range(n_samples):
                # 滑窗计算最小距离
                ts = train_data[sample_idx, dim_idx, :]
                min_dist = self._sliding_window_distance(ts, shapelet)
                distance_matrix[cand_idx, sample_idx] = min_dist
        
        # 使用决策树评估信息增益
        print(f"  计算信息增益...")
        for cand_idx in range(n_candidates):
            # 使用该距离作为特征训练决策树
            X = distance_matrix[cand_idx].reshape(-1, 1)
            dt = DecisionTreeClassifier(max_depth=1, random_state=42)
            dt.fit(X, train_labels)
            
            # 使用准确率作为信息增益的代理
            score = dt.score(X, train_labels)
            self.candidates_info[cand_idx, 3] = score
            
            # 填充标签信息（使用最常见的类别）
            predicted_labels = dt.predict(X)
            most_common_label = np.bincount(predicted_labels.astype(int)).argmax()
            self.candidates_info[cand_idx, 4] = most_common_label
        
        print(f"  ✓ 信息增益计算完成")
    
    def get_shapelet_info(self, number_of_shapelet: int = 3,
                         p: float = 0.1, pi: float = 0.1) -> np.ndarray:
        """
        获取按信息增益排序的 shapelet 信息
        
        Args:
            number_of_shapelet: 每类保留的 shapelet 数量
            p: 未使用
            pi: 未使用
            
        Returns:
            shapelet_info: 按类别和信息增益排序的 shapelet 信息
        """
        if len(self.candidates_info) == 0:
            return np.array([])
        
        # 按信息增益排序
        sorted_indices = np.argsort(self.candidates_info[:, 3])[::-1]
        sorted_info = self.candidates_info[sorted_indices]
        
        # 按类别分组
        unique_labels = np.unique(sorted_info[:, 4].astype(int))
        selected_info = []
        
        for label in unique_labels:
            label_mask = sorted_info[:, 4] == label
            label_info = sorted_info[label_mask]
            
            # 取前 number_of_shapelet 个
            n_to_take = min(number_of_shapelet, len(label_info))
            selected_info.append(label_info[:n_to_take])
        
        if len(selected_info) == 0:
            return np.array([])
        
        result = np.vstack(selected_info)
        print(f"  选择了 {len(result)} 个高质量 shapelet")
        
        return result
    
    def _sliding_window_distance(self, ts: np.ndarray, shapelet: np.ndarray) -> float:
        """
        计算时间序列与 shapelet 的最小滑窗距离（快速近似版本）
        
        Args:
            ts: 时间序列
            shapelet: shapelet
            
        Returns:
            min_distance: 最小欧氏距离（近似）
        """
        shapelet_len = len(shapelet)
        ts_len = len(ts)
        
        if ts_len < shapelet_len:
            return np.inf
        
        n_windows = ts_len - shapelet_len + 1
        
        # 只采样10个位置而非全部计算（大幅提速）
        sample_positions = np.linspace(0, n_windows-1, min(10, n_windows), dtype=int)
        
        min_dist = np.inf
        for pos in sample_positions:
            subsequence = ts[pos:pos+shapelet_len].copy()
            
            # Z-normalize
            subseq_mean = subsequence.mean()
            subseq_std = subsequence.std()
            if subseq_std > 1e-8:
                subsequence = (subsequence - subseq_mean) / subseq_std
            
            # 欧氏距离
            dist = np.linalg.norm(subsequence - shapelet)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist


# 为了兼容性，创建一个别名
mul_shapelet_discovery = ShapeletDiscover
