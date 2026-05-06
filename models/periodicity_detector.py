"""
周期检测与自适应异常评分模块
=============================
支持从正常样本自动检测周期性，并进行周期标准化的异常评分
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.fft import fft, fftfreq
from scipy import signal


class PeriodicityDetector:
    """
    周期性检测器
    
    从正常样本自动检测主要周期特性
    """
    
    def __init__(self, min_period: int = 10, max_period: int = None,
                 prominence_threshold: float = 0.1):
        """
        初始化周期检测器
        
        Args:
            min_period: 最小周期长度
            max_period: 最大周期长度（默认为序列长度/4）
            prominence_threshold: 峰值显著性阈值
        """
        self.min_period = min_period
        self.max_period = max_period
        self.prominence_threshold = prominence_threshold
        
        self.detected_periods = {}  # {scale: [period1, period2, ...]}
        self.primary_periods = {}   # {scale: primary_period}
    
    def detect_periods(self, distance_matrices_normal: List[Dict[int, np.ndarray]],
                      min_prominence: float = 0.01) -> Dict[int, List[int]]:
        """
        从正常样本的距离序列检测周期性
        
        Args:
            distance_matrices_normal: [{ws: [T, M_s]}, ...]
            min_prominence: FFT峰值最小显著性
        
        Returns:
            detected_periods: {scale: [period1, period2, ...]}
        """
        print(f"\n[PeriodicityDetector] 开始检测周期性...")
        
        detected = {}
        
        # 对每个尺度的距离序列进行周期检测
        scale_distances = {}
        for dist_mat_dict in distance_matrices_normal:
            for ws, dist_mat in dist_mat_dict.items():
                if ws not in scale_distances:
                    scale_distances[ws] = []
                # 聚合为单一异常分数序列（取最小值）
                scale_distance_seq = np.min(dist_mat, axis=1)  # [T]
                scale_distances[ws].append(scale_distance_seq)
        
        for ws, distance_seqs in scale_distances.items():
            # 合并多个样本的距离序列
            combined_seq = np.concatenate(distance_seqs)
            
            periods = self._detect_fft_periods(combined_seq, ws, min_prominence)
            detected[ws] = periods
            
            if len(periods) > 0:
                self.primary_periods[ws] = periods[0]
                print(f"  尺度 {ws}: 检测到周期 {periods}")
            else:
                self.primary_periods[ws] = None
                print(f"  尺度 {ws}: 未检测到明显周期")
        
        self.detected_periods = detected
        return detected
    
    def _detect_fft_periods(self, signal_seq: np.ndarray, scale: int,
                          min_prominence: float = 0.01) -> List[int]:
        """
        使用FFT检测周期性信号中的主要周期
        
        Args:
            signal_seq: [T] 信号序列
            scale: 该序列对应的时间窗口尺度（用于设置搜索范围）
            min_prominence: 峰值显著性阈值
        
        Returns:
            periods: 检测到的周期列表（按显著性排序）
        """
        T = len(signal_seq)
        
        # 标准化信号
        signal_normalized = (signal_seq - np.mean(signal_seq)) / (np.std(signal_seq) + 1e-8)
        
        # FFT计算功率谱
        fft_vals = np.abs(fft(signal_normalized))
        freqs = fftfreq(T)
        
        # 只关心正频率
        positive_freqs = freqs[:T//2]
        power = fft_vals[:T//2]
        
        # 搜索范围：周期长度在 [min_period, max_period]
        max_period = self.max_period if self.max_period is not None else T // 4
        min_freq = 1.0 / max_period
        max_freq = 1.0 / self.min_period
        
        # 提取搜索范围内的频率和功率
        valid_mask = (positive_freqs > min_freq) & (positive_freqs < max_freq)
        valid_freqs = positive_freqs[valid_mask]
        valid_power = power[valid_mask]
        
        if len(valid_power) == 0:
            return []
        
        # 找峰值
        peaks, properties = signal.find_peaks(valid_power, 
                                            prominence=np.max(valid_power) * min_prominence)
        
        if len(peaks) == 0:
            return []
        
        # 按显著性排序
        sorted_indices = np.argsort(properties['prominences'])[::-1]
        top_peaks = peaks[sorted_indices[:3]]  # 取top-3周期
        
        # 转换为周期长度
        periods = [int(np.round(1.0 / valid_freqs[p])) for p in top_peaks]
        periods = [p for p in periods if self.min_period <= p <= max_period]
        
        return periods
    
    def get_primary_period(self, scale: int) -> Optional[int]:
        """获取指定尺度的主要周期"""
        return self.primary_periods.get(scale)


class AdaptiveAnomalyScorer:
    """
    自适应异常评分器
    
    支持周期标准化和多时间尺度的异常评分组合
    """
    
    def __init__(self, aggregator=None, periodicity_detector=None):
        """
        初始化自适应评分器
        
        Args:
            aggregator: MultiScaleDistanceAggregator实例
            periodicity_detector: PeriodicityDetector实例
        """
        self.aggregator = aggregator
        self.periodicity_detector = periodicity_detector
        self.normal_baselines = {}  # {scale: [baseline_mean, baseline_std]}
    
    def learn_baselines(self, distance_matrices_normal: List[Dict[int, np.ndarray]]):
        """
        从正常样本学习每个时间位置的基准异常分数
        
        用于周期标准化时的参考
        
        Args:
            distance_matrices_normal: [{ws: [T, M_s]}, ...]
        """
        print(f"\n[AdaptiveAnomalyScorer] 学习正常样本基准...")
        
        scale_baselines = {}
        
        for dist_mat_dict in distance_matrices_normal:
            for ws, dist_mat in dist_mat_dict.items():
                if ws not in scale_baselines:
                    scale_baselines[ws] = []
                
                # 该尺度的异常分数序列（取最小值）
                score_seq = np.min(dist_mat, axis=1)  # [T]
                scale_baselines[ws].append(score_seq)
        
        # 计算周期统计
        for ws, score_seqs in scale_baselines.items():
            period = self.periodicity_detector.get_primary_period(ws) if self.periodicity_detector else None
            
            if period is not None and period > 0:
                # 周期标准化基准
                baseline_by_phase = self._compute_phase_baseline(
                    score_seqs, period
                )
                self.normal_baselines[ws] = baseline_by_phase
                print(f"  尺度 {ws}: 周期={period}, 相位基准已计算")
            else:
                # 全局基准
                combined = np.concatenate(score_seqs)
                self.normal_baselines[ws] = {
                    'global_mean': np.mean(combined),
                    'global_std': np.std(combined),
                }
                print(f"  尺度 {ws}: 全局基准已计算")
    
    def _compute_phase_baseline(self, score_seqs: List[np.ndarray],
                               period: int) -> Dict:
        """
        计算每个周期相位位置的基准统计
        
        返回dict: {phase_idx: {'mean': ..., 'std': ...}}
        """
        combined = np.concatenate(score_seqs)
        phase_stats = {}
        
        for phase in range(period):
            # 所有该相位的值
            phase_values = [combined[i] for i in range(len(combined)) if i % period == phase]
            
            if len(phase_values) > 0:
                phase_stats[phase] = {
                    'mean': np.mean(phase_values),
                    'std': np.std(phase_values),
                }
        
        return {'type': 'periodic', 'period': period, 'phases': phase_stats}
    
    def compute_anomaly_scores(self, distance_matrices: Dict[int, np.ndarray],
                              use_periodicity: bool = True) -> np.ndarray:
        """
        计算自适应异常评分
        
        Args:
            distance_matrices: {ws: [T, M_s]}
            use_periodicity: 是否使用周期信息
        
        Returns:
            anomaly_scores: [T] 时间序列异常评分
        """
        # 首先进行多尺度聚合
        if self.aggregator is not None:
            aggregated = self.aggregator.aggregate_distances(distance_matrices)
        else:
            # 降级：简单的最小值聚合
            all_distances = np.concatenate(list(distance_matrices.values()), axis=1)
            aggregated = np.min(all_distances, axis=1)
        
        if not use_periodicity or not self.periodicity_detector:
            return aggregated
        
        # 周期标准化处理
        T = len(aggregated)
        anomaly_scores = np.zeros(T)
        
        for t in range(T):
            # 多尺度周期标准化
            ws_scores = []
            
            for ws, dist_mat in distance_matrices.items():
                if t >= dist_mat.shape[0]:
                    continue
                
                dist_t = np.min(dist_mat[t, :])  # 该尺度、该时刻的距离
                
                if ws in self.normal_baselines:
                    baseline_info = self.normal_baselines[ws]
                    
                    if baseline_info.get('type') == 'periodic':
                        period = baseline_info['period']
                        phase = t % period
                        phase_stats = baseline_info['phases'].get(phase, {})
                        
                        mean = phase_stats.get('mean', 0)
                        std = phase_stats.get('std', 1)
                    else:
                        mean = baseline_info.get('global_mean', 0)
                        std = baseline_info.get('global_std', 1)
                    
                    # Z-score标准化
                    if std > 1e-6:
                        norm_score = (dist_t - mean) / std
                    else:
                        norm_score = dist_t - mean
                    
                    ws_scores.append(norm_score)
            
            # 多尺度评分聚合
            if ws_scores:
                anomaly_scores[t] = np.mean(ws_scores)
            else:
                anomaly_scores[t] = aggregated[t]
        
        return anomaly_scores
    
    def compute_temporal_anomaly_scores(self, distance_matrices: Dict[int, np.ndarray],
                                       temporal_window: int = 5,
                                       use_periodicity: bool = True) -> np.ndarray:
        """
        考虑时间邻域的多时间尺度异常评分
        
        Args:
            distance_matrices: {ws: [T, M_s]}
            temporal_window: 时间邻域窗口大小（前后各temporal_window步）
            use_periodicity: 是否使用周期信息
        
        Returns:
            temporal_scores: [T] 时间序列异常评分
        """
        base_scores = self.compute_anomaly_scores(distance_matrices, use_periodicity)
        T = len(base_scores)
        
        temporal_scores = np.zeros(T)
        
        for t in range(T):
            # 收集时间邻域内的评分
            start = max(0, t - temporal_window)
            end = min(T, t + temporal_window + 1)
            
            neighborhood_scores = base_scores[start:end]
            
            # 加权组合（中心时刻权重最高）
            distances = np.abs(np.arange(start, end) - t)
            weights = 1.0 / (1.0 + distances)
            weights = weights / np.sum(weights)
            
            temporal_scores[t] = np.dot(neighborhood_scores, weights)
        
        return temporal_scores
