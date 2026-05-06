"""
多尺度多通道距离聚合模块
========================
支持多个时间窗口尺度的Shapelet距离聚合与权重学习
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum


class AggregationMethod(Enum):
    """距离聚合方法"""
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MEDIAN = "median"


class MultiScaleDistanceAggregator:
    """
    多尺度距离聚合器
    
    整合多个window_size的Shapelet距离，生成单一的时间序列异常分数
    """
    
    def __init__(self, window_sizes: List[int] = None, 
                 aggregation_method: str = "weighted_mean",
                 learn_weights: bool = True):
        """
        初始化多尺度聚合器
        
        Args:
            window_sizes: 多个窗口尺度 [10, 20, 30, 40]
            aggregation_method: 聚合方式 (min/max/mean/weighted_mean/median)
            learn_weights: 是否学习尺度权重
        """
        if window_sizes is None:
            window_sizes = [10, 20, 30, 40]
        
        self.window_sizes = sorted(window_sizes)
        self.n_scales = len(self.window_sizes)
        self.aggregation_method = aggregation_method
        self.learn_weights = learn_weights
        
        # 尺度权重（初始均匀）
        self.scale_weights = np.ones(self.n_scales) / self.n_scales
        
        print(f"\n[MultiScaleDistanceAggregator] 初始化完成")
        print(f"  - 时间尺度: {self.window_sizes}")
        print(f"  - 聚合方法: {aggregation_method}")
        print(f"  - 学习权重: {learn_weights}")
    
    def aggregate_distances(self, distance_matrices: Dict[int, np.ndarray],
                          method: str = None) -> np.ndarray:
        """
        聚合多个尺度的距离矩阵为单一异常分数序列
        
        Args:
            distance_matrices: {window_size: [T, M_s]} 距离矩阵字典
                T: 时间步数
                M_s: 该尺度的Shapelet数量
            method: 聚合方法（如不指定则使用初始化时的方法）
        
        Returns:
            aggregated: [T] - 聚合后的异常分数
        """
        if method is None:
            method = self.aggregation_method
        
        # 获取时间长度（应该对所有尺度相同）
        seq_lengths = [distance_matrices[ws].shape[0] 
                      for ws in self.window_sizes 
                      if ws in distance_matrices]
        
        if not seq_lengths:
            raise ValueError("没有找到有效的距离矩阵")
        
        T = seq_lengths[0]
        
        # 为每个时间步聚合
        aggregated = np.zeros(T)
        
        for t in range(T):
            scale_scores = []
            
            for scale_idx, ws in enumerate(self.window_sizes):
                if ws not in distance_matrices:
                    continue
                
                dist_matrix = distance_matrices[ws]  # [T, M_s]
                
                if t >= dist_matrix.shape[0]:
                    continue
                
                # 该时间步下所有Shapelet的距离
                distances_at_t = dist_matrix[t, :]  # [M_s]
                
                # 按方法聚合该尺度的距离
                if method == AggregationMethod.MIN.value or method == "min":
                    scale_score = np.min(distances_at_t)
                elif method == AggregationMethod.MAX.value or method == "max":
                    scale_score = np.max(distances_at_t)
                elif method == AggregationMethod.MEDIAN.value or method == "median":
                    scale_score = np.median(distances_at_t)
                else:  # mean 或 weighted_mean
                    scale_score = np.mean(distances_at_t)
                
                scale_scores.append(scale_score)
            
            # 用权重聚合各尺度分数
            scale_scores = np.array(scale_scores)
            
            if method == AggregationMethod.WEIGHTED_MEAN.value or method == "weighted_mean":
                aggregated[t] = np.dot(scale_scores, self.scale_weights[:len(scale_scores)])
            else:
                aggregated[t] = np.mean(scale_scores)
        
        return aggregated
    
    def learn_scale_weights(self, distance_matrices_normal: List[Dict[int, np.ndarray]],
                           distance_matrices_anomaly: List[Dict[int, np.ndarray]]) -> np.ndarray:
        """
        从正常和异常样本学习尺度权重
        
        思路：优先给能更好区分正常和异常的尺度赋予更高权重
        
        Args:
            distance_matrices_normal: 正常样本的距离矩阵列表 [{ws: [T, M_s]}, ...]
            distance_matrices_anomaly: 异常样本的距离矩阵列表
        
        Returns:
            weights: [n_scales] 学习得到的权重
        """
        if not self.learn_weights:
            return self.scale_weights
        
        # 对每个尺度计算区分能力（基于方差比）
        scale_scores = np.zeros(self.n_scales)
        
        for scale_idx, ws in enumerate(self.window_sizes):
            # 正常样本在该尺度的距离分布
            normal_distances = []
            for dist_mat_dict in distance_matrices_normal:
                if ws in dist_mat_dict:
                    normal_distances.extend(dist_mat_dict[ws].flatten())
            
            # 异常样本在该尺度的距离分布
            anomaly_distances = []
            for dist_mat_dict in distance_matrices_anomaly:
                if ws in dist_mat_dict:
                    anomaly_distances.extend(dist_mat_dict[ws].flatten())
            
            if len(normal_distances) == 0 or len(anomaly_distances) == 0:
                scale_scores[scale_idx] = 1.0
                continue
            
            normal_distances = np.array(normal_distances)
            anomaly_distances = np.array(anomaly_distances)
            
            # 计算区分能力：异常平均距离 / 正常平均距离
            normal_mean = np.mean(normal_distances)
            anomaly_mean = np.mean(anomaly_distances)
            
            if normal_mean > 1e-6:
                scale_scores[scale_idx] = anomaly_mean / normal_mean
            else:
                scale_scores[scale_idx] = 1.0
        
        # 归一化权重
        self.scale_weights = scale_scores / np.sum(scale_scores)
        
        print(f"\n[MultiScaleDistanceAggregator] 已学习尺度权重:")
        for ws, w in zip(self.window_sizes, self.scale_weights):
            print(f"  窗口大小 {ws:2d}: {w:.4f}")
        
        return self.scale_weights
    
    def get_scale_weights(self) -> np.ndarray:
        """返回当前尺度权重"""
        return self.scale_weights
    
    def set_scale_weights(self, weights: np.ndarray):
        """手动设置尺度权重"""
        if len(weights) != self.n_scales:
            raise ValueError(f"权重长度应为{self.n_scales}")
        
        self.scale_weights = weights / np.sum(weights)
        print(f"\n[MultiScaleDistanceAggregator] 已设置尺度权重:")
        for ws, w in zip(self.window_sizes, self.scale_weights):
            print(f"  窗口大小 {ws:2d}: {w:.4f}")


class DistanceNormalizer:
    """
    距离标准化器
    
    使用正常样本的分布对距离进行标准化，便于后续异常评分
    """
    
    def __init__(self):
        self.normal_stats = {}  # {scale: {'mean': ..., 'std': ...}}
    
    def learn_statistics(self, distance_matrices_normal: List[Dict[int, np.ndarray]]):
        """
        从正常样本学习距离统计
        
        Args:
            distance_matrices_normal: [{ws: [T, M_s]}, ...]
        """
        all_distances = {}
        
        for dist_mat_dict in distance_matrices_normal:
            for ws, dist_mat in dist_mat_dict.items():
                if ws not in all_distances:
                    all_distances[ws] = []
                all_distances[ws].extend(dist_mat.flatten())
        
        for ws, distances in all_distances.items():
            distances = np.array(distances)
            self.normal_stats[ws] = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'median': np.median(distances),
                'q25': np.percentile(distances, 25),
                'q75': np.percentile(distances, 75),
            }
        
        print(f"\n[DistanceNormalizer] 已学习正常样本统计:")
        for ws, stats in self.normal_stats.items():
            print(f"  窗口 {ws}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    def normalize(self, distance_matrices: Dict[int, np.ndarray], 
                 method: str = "zscore") -> Dict[int, np.ndarray]:
        """
        标准化距离矩阵
        
        Args:
            distance_matrices: {ws: [T, M_s]}
            method: 标准化方法 (zscore/minmax/robust)
        
        Returns:
            normalized_matrices: 标准化后的距离矩阵
        """
        normalized = {}
        
        for ws, dist_mat in distance_matrices.items():
            if ws not in self.normal_stats:
                # 没有正常样本统计，直接返回
                normalized[ws] = dist_mat
                continue
            
            stats = self.normal_stats[ws]
            
            if method == "zscore":
                if stats['std'] > 1e-6:
                    normalized[ws] = (dist_mat - stats['mean']) / stats['std']
                else:
                    normalized[ws] = dist_mat - stats['mean']
            
            elif method == "minmax":
                # 使用Q25和Q75作为边界
                q_range = stats['q75'] - stats['q25']
                if q_range > 1e-6:
                    normalized[ws] = (dist_mat - stats['q25']) / q_range
                else:
                    normalized[ws] = dist_mat
            
            elif method == "robust":
                # 用中位数和IQR进行Robust标准化
                q_range = stats['q75'] - stats['q25']
                if q_range > 1e-6:
                    normalized[ws] = (dist_mat - stats['median']) / q_range
                else:
                    normalized[ws] = dist_mat
            
            else:
                normalized[ws] = dist_mat
        
        return normalized
