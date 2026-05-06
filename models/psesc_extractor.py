"""
PseSC 特征提取器
===============
将隶属度序列转换为紧凑的特征向量

数学定义：
---------
位置归一化: p_t = (t - 0.5) / T
总隶属度: S = Σ_{t=1}^T μ_t
数值稳健项: ε = 1e-8

时间统计量:
c1 = Σ_t (p_t * μ_t) / S              # 质心
c2 = Σ_t (p_t - c1)^2 * μ_t / S       # 方差
c3 = Σ_t (p_t - c1)^3 * μ_t / ((c2 + ε)^3 * S)  # 偏度

顺序统计量:
θ1 ≥ θ2 ≥ θ3  # {μ_t} 的前三大值

PseSC 表征:
PseSC_time(s) = [c1, c2, c3, θ1, θ2, θ3]^T ∈ R^6

数值兜底:
若 S < 1e-6，取 c1 = 0.5, c2 = c3 = 0, θ_{1..3} = 0
"""

import numpy as np
from typing import Dict, List, Tuple


class PseSCExtractor:
    """
    PseSC-Time 特征提取器
    
    将隶属度序列 μ(t) 转换为 6 维特征向量
    """
    
    def __init__(self, n_order_stats: int = 3, eps: float = 1e-8, 
                 zero_threshold: float = 1e-6):
        """
        初始化提取器
        
        Args:
            n_order_stats: 顺序统计量个数（默认 3: top-3）
            eps: 数值稳健项
            zero_threshold: 零值判断阈值
        """
        self.n_order = n_order_stats
        self.eps = eps
        self.zero_threshold = zero_threshold
        self.feature_dim = 3 + n_order_stats  # c1, c2, c3 + θ1, θ2, θ3
        
        print(f"[PseSCExtractor] 初始化完成")
        print(f"  - 时间统计量: 3 (c1, c2, c3)")
        print(f"  - 顺序统计量: {n_order_stats}")
        print(f"  - 总特征维度: {self.feature_dim}")
        print(f"  - 数值稳健项 ε: {eps}")
        print(f"  - 零值阈值: {zero_threshold}")
    
    def extract_shapelet_psesc(self, memberships: np.ndarray, 
                              positions: np.ndarray) -> np.ndarray:
        """
        提取单个 shapelet 的 PseSC 特征
        
        Args:
            memberships: (T,) 隶属度序列
            positions: (T,) 位置索引
            
        Returns:
            psesc: (6,) [c1, c2, c3, θ1, θ2, θ3]
        """
        T = len(memberships)
        
        if T == 0:
            # 空序列
            return self._get_zero_feature()
        
        # 计算总隶属度
        S = np.sum(memberships)
        
        # 数值兜底
        if S < self.zero_threshold:
            return self._get_zero_feature()
        
        # 位置归一化: p_t = (t - 0.5) / T
        p_t = (positions + 0.5) / T  # positions 从 0 开始，所以 +0.5
        
        # 时间统计量
        # c1: 质心（加权平均位置）
        c1 = np.sum(p_t * memberships) / S
        
        # c2: 方差（分布宽度）
        c2 = np.sum((p_t - c1) ** 2 * memberships) / S
        
        # c3: 偏度（分布对称性）
        # 公式: Σ_t (p_t - c1)^3 * μ_t / ((c2 + ε)^3 * S)
        c3_numerator = np.sum((p_t - c1) ** 3 * memberships)
        c3_denominator = ((c2 + self.eps) ** 1.5) * S
        c3 = c3_numerator / c3_denominator
        
        # 顺序统计量: θ1 ≥ θ2 ≥ θ3
        sorted_memberships = np.sort(memberships)[::-1]  # 降序
        theta = np.zeros(self.n_order)
        
        for i in range(min(self.n_order, len(sorted_memberships))):
            theta[i] = sorted_memberships[i]
        
        # 拼接特征
        psesc = np.concatenate([
            [c1, c2, c3],
            theta
        ])
        
        return psesc
    
    def _get_zero_feature(self) -> np.ndarray:
        """
        数值兜底：返回零特征
        
        c1 = 0.5, c2 = c3 = 0, θ_{1..3} = 0
        """
        zero_feature = np.zeros(self.feature_dim)
        zero_feature[0] = 0.5  # c1 = 0.5
        return zero_feature
    
    def extract_class_psesc(self, class_memberships: Dict) -> np.ndarray:
        """
        提取单个类别所有 shapelet 的 PseSC 特征
        
        Args:
            class_memberships: {
                'shapelet_0': {...},
                'shapelet_1': {...},
                ...
            }
            
        Returns:
            class_psesc: (n_shapelets * feature_dim,)
        """
        shapelet_features = []
        
        for shapelet_key in sorted(class_memberships.keys()):
            shapelet_data = class_memberships[shapelet_key]
            memberships = shapelet_data['memberships']
            positions = shapelet_data['positions']
            
            psesc = self.extract_shapelet_psesc(memberships, positions)
            shapelet_features.append(psesc)
        
        if len(shapelet_features) == 0:
            return np.array([])
        
        return np.concatenate(shapelet_features)
    
    def extract_full_psesc(self, memberships_dict: Dict) -> np.ndarray:
        """
        提取完整的 PseSC 特征向量
        
        Args:
            memberships_dict: {
                'class_0': {...},
                'class_1': {...},
                ...
            }
            
        Returns:
            full_psesc: (n_classes * n_shapelets * feature_dim,)
        """
        all_features = []
        
        for class_key in sorted(memberships_dict.keys()):
            class_memberships = memberships_dict[class_key]
            class_psesc = self.extract_class_psesc(class_memberships)
            
            if len(class_psesc) > 0:
                all_features.append(class_psesc)
        
        if len(all_features) == 0:
            return np.array([])
        
        return np.concatenate(all_features)
    
    def extract_batch_psesc(self, memberships_list: List[Dict]) -> np.ndarray:
        """
        批量提取 PseSC 特征
        
        Args:
            memberships_list: [memberships_dict_1, memberships_dict_2, ...]
            
        Returns:
            batch_psesc: (n_samples, feature_dim)
        """
        batch_features = []
        
        for memberships_dict in memberships_list:
            full_psesc = self.extract_full_psesc(memberships_dict)
            batch_features.append(full_psesc)
        
        return np.array(batch_features)
    
    def get_feature_names(self, n_classes: int, n_per_class: int) -> List[str]:
        """
        获取特征名称（用于解释）
        
        Args:
            n_classes: 类别数
            n_per_class: 每类 shapelet 数
            
        Returns:
            feature_names: 特征名称列表
        """
        names = []
        
        for class_idx in range(n_classes):
            for shapelet_idx in range(n_per_class):
                prefix = f"C{class_idx}_S{shapelet_idx}"
                names.extend([
                    f"{prefix}_c1_centroid",
                    f"{prefix}_c2_variance",
                    f"{prefix}_c3_skewness",
                ])
                for k in range(self.n_order):
                    names.append(f"{prefix}_theta{k+1}")
        
        return names


def extract_psesc_features(memberships_dict: Dict, 
                          n_order_stats: int = 3) -> np.ndarray:
    """
    便捷函数：提取 PseSC 特征
    
    Args:
        memberships_dict: 隶属度字典
        n_order_stats: 顺序统计量个数
        
    Returns:
        psesc_features: PseSC 特征向量
    """
    extractor = PseSCExtractor(n_order_stats=n_order_stats)
    return extractor.extract_full_psesc(memberships_dict)


def diagnose_psesc_features(psesc_features: np.ndarray, 
                           feature_names: List[str] = None) -> Dict:
    """
    诊断 PseSC 特征质量
    
    Args:
        psesc_features: (n_samples, feature_dim) 或 (feature_dim,)
        feature_names: 特征名称列表
        
    Returns:
        diagnostics: 诊断信息字典
    """
    if psesc_features.ndim == 1:
        psesc_features = psesc_features.reshape(1, -1)
    
    n_samples, feature_dim = psesc_features.shape
    
    # 计算统计量
    feature_means = np.mean(psesc_features, axis=0)
    feature_stds = np.std(psesc_features, axis=0)
    feature_mins = np.min(psesc_features, axis=0)
    feature_maxs = np.max(psesc_features, axis=0)
    
    # 零值特征检测
    zero_features = np.sum(np.abs(psesc_features) < 1e-10, axis=0)
    zero_ratio = zero_features / n_samples
    
    # 恒定特征检测
    constant_features = feature_stds < 1e-10
    
    diagnostics = {
        'n_samples': n_samples,
        'feature_dim': feature_dim,
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'feature_ranges': feature_maxs - feature_mins,
        'zero_ratio': zero_ratio,
        'n_constant_features': np.sum(constant_features),
        'constant_features': constant_features,
    }
    
    if feature_names is not None:
        diagnostics['feature_names'] = feature_names
        diagnostics['constant_feature_names'] = [
            feature_names[i] for i in range(len(constant_features))
            if constant_features[i]
        ]
    
    return diagnostics


if __name__ == "__main__":
    # 测试示例
    print("PseSC 特征提取器测试")
    print("=" * 50)
    
    # 模拟隶属度序列
    np.random.seed(42)
    T = 80  # 滑窗数
    
    memberships = np.random.rand(T) * 0.5 + 0.2
    positions = np.arange(T)
    
    # 提取特征
    extractor = PseSCExtractor(n_order_stats=3)
    psesc = extractor.extract_shapelet_psesc(memberships, positions)
    
    print(f"\n单个 shapelet 的 PseSC 特征:")
    print(f"  - c1 (质心): {psesc[0]:.4f}")
    print(f"  - c2 (方差): {psesc[1]:.4f}")
    print(f"  - c3 (偏度): {psesc[2]:.4f}")
    print(f"  - θ1: {psesc[3]:.4f}")
    print(f"  - θ2: {psesc[4]:.4f}")
    print(f"  - θ3: {psesc[5]:.4f}")
    
    # 测试零值兜底
    print(f"\n测试零值兜底:")
    zero_memberships = np.zeros(T)
    zero_psesc = extractor.extract_shapelet_psesc(zero_memberships, positions)
    print(f"  零隶属度序列的 PseSC: {zero_psesc}")
    
    # 模拟完整 memberships_dict
    memberships_dict = {
        'class_0': {
            'shapelet_0': {'memberships': memberships, 'positions': positions},
            'shapelet_1': {'memberships': memberships * 0.8, 'positions': positions},
        },
        'class_1': {
            'shapelet_0': {'memberships': memberships * 1.2, 'positions': positions},
            'shapelet_1': {'memberships': memberships * 0.6, 'positions': positions},
        }
    }
    
    full_psesc = extractor.extract_full_psesc(memberships_dict)
    print(f"\n完整 PseSC 特征:")
    print(f"  - 维度: {len(full_psesc)}")
    print(f"  - 期望维度: 2 类 × 2 shapelet/类 × 6 = 24")
    print(f"  - 范围: [{np.min(full_psesc):.4f}, {np.max(full_psesc):.4f}]")
    
    # 特征诊断
    print(f"\n特征诊断:")
    diag = diagnose_psesc_features(full_psesc)
    print(f"  - 样本数: {diag['n_samples']}")
    print(f"  - 特征维度: {diag['feature_dim']}")
    print(f"  - 恒定特征数: {diag['n_constant_features']}")
