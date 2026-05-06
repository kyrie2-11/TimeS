"""
改进的多变量时序异常检测器
==========================
整合多维Shapelet、多尺度距离聚合、周期检测的完整异常检测系统
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List

from Shapelet.multivariate_shapelet_discovery import MultivariateShapeletDiscover
from models.membership_mapping import MembershipMapper
from models.distance_aggregation import MultiScaleDistanceAggregator, DistanceNormalizer
from models.periodicity_detector import PeriodicityDetector, AdaptiveAnomalyScorer


class MultiVariateAnomalyDetector:
    """
    改进的多变量时序异常检测器
    
    核心创新：
    1. 多维联合Shapelet提取（跨维度相似性）
    2. 多尺度多通道距离聚合
    3. 周期感知的自适应异常评分
    """
    
    def __init__(self, window_sizes: List[int] = None,
                 n_shapelets_per_scale: int = 20,
                 detection_method: str = 'iforest',
                 contamination: float = 0.1,
                 use_periodicity: bool = True,
                 use_temporal_window: bool = True,
                 temporal_window_size: int = 5):
        """
        初始化异常检测器
        
        Args:
            window_sizes: 多个Shapelet窗口尺度 [10, 20, 30, 40]
            n_shapelets_per_scale: 每个尺度的Shapelet数量
            detection_method: 异常检测方法 (iforest/svm/elliptic)
            contamination: 异常比例估计
            use_periodicity: 是否使用周期检测
            use_temporal_window: 是否使用时间邻域加权
            temporal_window_size: 时间窗口大小
        """
        if window_sizes is None:
            window_sizes = [10, 20, 30, 40]
        
        self.window_sizes = window_sizes
        self.n_shapelets_per_scale = n_shapelets_per_scale
        self.detection_method = detection_method
        self.contamination = contamination
        self.use_periodicity = use_periodicity
        self.use_temporal_window = use_temporal_window
        self.temporal_window_size = temporal_window_size
        
        # 核心组件
        self.shapelet_discoverers = {}  # {window_size: ShapeletDiscover}
        self.membership_mappers = {}    # {window_size: MembershipMapper}
        self.distance_aggregator = MultiScaleDistanceAggregator(
            window_sizes=window_sizes,
            aggregation_method="weighted_mean",
            learn_weights=True
        )
        self.distance_normalizer = DistanceNormalizer()
        self.periodicity_detector = PeriodicityDetector() if use_periodicity else None
        self.anomaly_scorer = AdaptiveAnomalyScorer(
            aggregator=self.distance_aggregator,
            periodicity_detector=self.periodicity_detector
        )
        
        # 异常检测模型
        self.anomaly_model = None
        self.scaler = StandardScaler()
        self.threshold = None
        
        self.is_fitted = False
        
        print(f"\n{'='*70}")
        print(f"[MultiVariateAnomalyDetector] 初始化完成")
        print(f"{'='*70}")
        print(f"时间尺度: {self.window_sizes}")
        print(f"检测方法: {detection_method}")
        print(f"周期检测: {use_periodicity}")
        print(f"时间邻域: {use_temporal_window}")
    
    def fit(self, X_normal: np.ndarray):
        """
        用正常样本训练异常检测器
        
        Args:
            X_normal: [seq_len, n_dims] 正常样本（单个多变量时间序列）
        """
        print(f"\n{'='*70}")
        print(f"[异常检测器] 开始训练...")
        print(f"{'='*70}")
        
        seq_len, n_dims = X_normal.shape
        
        # 数据格式转换：[seq_len, n_dims] -> [1, n_dims, seq_len]
        X_train_3d = X_normal.T.reshape(1, n_dims, seq_len)
        y_train = np.array([1])  # 全部标记为"正常"（虚拟标签）
        
        print(f"输入数据形状: {X_normal.shape} (seq_len, n_dims)")
        print(f"重构为: {X_train_3d.shape} (n_samples=1, n_dims, seq_len)")
        
        # ========== 多尺度Shapelet提取 ==========
        print(f"\n[第1步] 多尺度Shapelet提取...")
        
        all_distance_matrices = []  # 用于学习距离聚合权重
        
        for ws in self.window_sizes:
            print(f"\n  处理窗口大小: {ws}")
            
            # 创建Shapelet发现器
            discoverer = MultivariateShapeletDiscover(
                window_size=ws,
                num_pip=0.2,
                univariate_ratio=0.7,
                multivariate_ratio=0.3,
                max_multivariate_dims=3
            )
            
            # 提取候选Shapelet
            discoverer.extract_candidate(X_train_3d)
            
            # 评估候选Shapelet（虽然只有1个类，但仍需完成流程）
            discoverer.discovery(X_train_3d, y_train)
            
            # 选择最优Shapelet
            shapelets_dict = discoverer.get_shapelets_by_class(
                n_per_class=self.n_shapelets_per_scale
            )
            
            self.shapelet_discoverers[ws] = discoverer
            
            print(f"  ✓ 提取了 {len(discoverer.candidates)} 候选, 选择了 {self.n_shapelets_per_scale} 个")
        
        # ========== 多尺度隶属度计算 ==========
        print(f"\n[第2步] 计算隶属度映射...")
        
        distance_matrices_normal = []  # 用于学习异常评分基准
        
        for ws in self.window_sizes:
            discoverer = self.shapelet_discoverers[ws]
            shapelets_dict = discoverer.get_shapelets_by_class(
                n_per_class=self.n_shapelets_per_scale
            )
            
            # 获取选定的Shapelet
            # 修复：获取所有可用的类别的Shapelet（因为可能不是标签0）
            selected_indices = []
            for label_key in sorted(shapelets_dict.keys()):
                selected_indices.extend(shapelets_dict[label_key])
            
            if len(selected_indices) == 0:
                print(f"  警告: 窗口大小{ws}没有选定Shapelet")
                continue
            
            shapelets = discoverer.get_selected_shapelets(selected_indices)
            
            # 计算该窗口下所有Shapelet与样本的距离矩阵
            T = seq_len - ws + 1
            M_s = len(shapelets)
            dist_matrix = np.zeros((T, M_s))

            for m_idx, s_info in enumerate(shapelets):
                shapelet = s_info['shapelet']
                dims = s_info['dims']
                
                for t in range(min(T, 20)):  # 限制计算范围以加快速度
                    # 获取窗口内的数据
                    window = X_normal[t:t+ws, :]  # [ws, n_dims]
                    window_3d = window.T.reshape(1, n_dims, ws)
                    
                    # 计算多维距离
                    dist = discoverer.multivariate_distance(
                        window_3d[0], shapelet, dims, 
                        weights=discoverer.channel_weights
                    )
                    dist_matrix[t, m_idx] = dist
            
            distance_matrices_normal.append({ws: dist_matrix})
            print(f"  ✓ 窗口{ws}: 距离矩阵形状 {dist_matrix.shape}")
        
        # ========== 学习距离聚合权重 ==========
        print(f"\n[第3步] 学习多尺度权重...")
        
        # 为学习权重，需要分离正常和"异常"样本
        # 这里简单地将序列的前70%作为正常，后30%作为异常样本（用于权重学习）
        split_idx = int(seq_len * 0.7)
        
        dist_normal = [d for i, d in enumerate(distance_matrices_normal) 
                      if all(d[ws].shape[0] >= split_idx for ws in d.keys())]
        dist_anomaly = [{ws: d[ws][split_idx:, :] for ws in d.keys()} 
                       for d in distance_matrices_normal]
        
        if dist_normal and dist_anomaly:
            self.distance_aggregator.learn_scale_weights(dist_normal, dist_anomaly)
        else:
            print(f"  警告: 权重学习数据不足，使用默认权重")
        
        # ========== 学习距离标准化统计 ==========
        print(f"\n[第4步] 学习距离统计...")
        self.distance_normalizer.learn_statistics(distance_matrices_normal)
        
        # ========== 周期检测 ==========
        if self.use_periodicity:
            print(f"\n[第5步] 检测周期性...")
            self.periodicity_detector.detect_periods(distance_matrices_normal)
            self.anomaly_scorer.learn_baselines(distance_matrices_normal)
        
        # ========== 异常检测模型训练 ==========
        print(f"\n[第6步] 训练异常检测模型...")
        
        # 使用全部样本的异常分数训练检测器
        if self.use_temporal_window:
            anomaly_scores = self.anomaly_scorer.compute_temporal_anomaly_scores(
                {ws: d[ws] for d in distance_matrices_normal for ws in d.keys()},
                temporal_window=self.temporal_window_size,
                use_periodicity=self.use_periodicity
            )
        else:
            # 合并所有距离矩阵
            combined_dist_mats = {}
            for dist_dict in distance_matrices_normal:
                combined_dist_mats.update(dist_dict)
            
            anomaly_scores = self.anomaly_scorer.compute_anomaly_scores(
                combined_dist_mats,
                use_periodicity=self.use_periodicity
            )
        
        # 标准化异常分数
        X_scores = anomaly_scores.reshape(-1, 1)
        X_scores = self.scaler.fit_transform(X_scores)
        
        # 训练异常检测模型
        if self.detection_method == 'iforest':
            self.anomaly_model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.detection_method == 'svm':
            self.anomaly_model = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto'
            )
        elif self.detection_method == 'elliptic':
            self.anomaly_model = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        else:
            raise ValueError(f"未知的检测方法: {self.detection_method}")
        
        self.anomaly_model.fit(X_scores)
        
        # 计算决策阈值
        decision_scores = self.anomaly_model.decision_function(X_scores)
        self.threshold = np.percentile(decision_scores, (1 - self.contamination) * 100)
        
        print(f"  ✓ 模型训练完成")
        print(f"  决策阈值: {self.threshold:.4f}")
        print(f"  异常分数范围: [{np.min(anomaly_scores):.4f}, {np.max(anomaly_scores):.4f}]")
        
        self.is_fitted = True
        print(f"\n✓ 异常检测器训练完成！")
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测异常
        
        Args:
            X_test: [seq_len, n_dims] 测试数据
        
        Returns:
            predictions: [seq_len] 异常标签 (1=异常, 0=正常)
            anomaly_scores: [seq_len] 异常分数（越高越异常）
        """
        if not self.is_fitted:
            raise RuntimeError("检测器未训练，请先调用fit()")
        
        print(f"\n[异常检测器] 开始预测...")
        
        seq_len, n_dims = X_test.shape
        
        # 多尺度距离矩阵
        distance_matrices = {}
        
        for ws in self.window_sizes:
            discoverer = self.shapelet_discoverers[ws]
            shapelets_dict = discoverer.get_shapelets_by_class(
                n_per_class=self.n_shapelets_per_scale
            )
            
            selected_indices = shapelets_dict.get(0, [])
            if len(selected_indices) == 0:
                continue
            
            shapelets = discoverer.get_selected_shapelets(selected_indices)
            
            T = seq_len - ws + 1
            M_s = len(shapelets)
            dist_matrix = np.zeros((T, M_s))
            
            for m_idx, s_info in enumerate(shapelets):
                shapelet = s_info['shapelet']
                dims = s_info['dims']
                
                for t in range(T):
                    window = X_test[t:t+ws, :]
                    window_3d = window.T.reshape(1, n_dims, ws)
                    dist = discoverer.multivariate_distance(
                        window_3d[0], shapelet, dims,
                        weights=discoverer.channel_weights
                    )
                    dist_matrix[t, m_idx] = dist
            
            distance_matrices[ws] = dist_matrix
        
        # 计算异常分数
        if self.use_temporal_window:
            anomaly_scores = self.anomaly_scorer.compute_temporal_anomaly_scores(
                distance_matrices,
                temporal_window=self.temporal_window_size,
                use_periodicity=self.use_periodicity
            )
        else:
            anomaly_scores = self.anomaly_scorer.compute_anomaly_scores(
                distance_matrices,
                use_periodicity=self.use_periodicity
            )
        
        # 标准化
        X_scores = anomaly_scores.reshape(-1, 1)
        X_scores = self.scaler.transform(X_scores)
        
        # 异常检测
        predictions = self.anomaly_model.predict(X_scores)
        # OneClassSVM返回1表示inlier, -1表示outlier，需要转换
        if self.detection_method == 'svm':
            predictions = (predictions == -1).astype(int)
        else:
            # IForest和Elliptic返回-1表示异常，1表示正常
            predictions = (predictions == -1).astype(int)
        
        print(f"  预测完成: 异常样本数={predictions.sum()}/{len(predictions)}")
        
        return predictions, anomaly_scores
    
    def predict_with_details(self, X_test: np.ndarray) -> Dict:
        """
        预测异常并返回详细信息
        
        Returns:
            dict: {
                'predictions': [T],
                'anomaly_scores': [T],
                'distance_matrices': {ws: [T, M_s]},
                'scale_weights': [...],
                'primary_periods': {...}
            }
        """
        predictions, anomaly_scores = self.predict(X_test)
        
        return {
            'predictions': predictions,
            'anomaly_scores': anomaly_scores,
            'scale_weights': self.distance_aggregator.get_scale_weights(),
            'primary_periods': self.periodicity_detector.primary_periods if self.periodicity_detector else None,
            'window_sizes': self.window_sizes,
            'detection_method': self.detection_method,
            'contamination': self.contamination,
        }
