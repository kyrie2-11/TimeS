"""
可视化工具模块
============
提供隶属度和 PseSC 特征的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_membership_heatmap(time_series: np.ndarray,
                            memberships_dict: Dict,
                            true_label: Optional[int] = None,
                            predicted_label: Optional[int] = None,
                            output_path: Optional[str] = None,
                            figsize: tuple = (12, 8)):
    """
    绘制隶属度热力图
    
    Args:
        time_series: (n_dims, seq_len) 原始时间序列
        memberships_dict: 隶属度字典
        true_label: 真实标签
        predicted_label: 预测标签
        output_path: 输出文件路径
        figsize: 图像大小
    """
    n_classes = len(memberships_dict)
    
    fig, axes = plt.subplots(n_classes + 1, 1, figsize=figsize,
                            gridspec_kw={'height_ratios': [1] + [2]*n_classes})
    
    # 顶部：原始时间序列
    if time_series.shape[0] < time_series.shape[1]:
        time_series = time_series.T
    
    seq_len = time_series.shape[0]
    t_axis = np.arange(seq_len)
    
    for dim in range(time_series.shape[1]):
        axes[0].plot(t_axis, time_series[:, dim], alpha=0.6, linewidth=0.8)
    
    axes[0].set_title('原始时间序列（多维度叠加）', fontsize=12, pad=10)
    axes[0].set_xlim(0, seq_len)
    axes[0].grid(alpha=0.3)
    
    # 每个类别一行热力图
    for class_idx in range(n_classes):
        class_key = f'class_{class_idx}'
        class_memberships = memberships_dict[class_key]
        
        n_shapelets = len(class_memberships)
        
        # 构建热力图数据
        max_len = max(len(class_memberships[sk]['memberships']) 
                     for sk in class_memberships.keys())
        
        heatmap_data = np.zeros((n_shapelets, max_len))
        
        for i, shapelet_key in enumerate(sorted(class_memberships.keys())):
            memberships = class_memberships[shapelet_key]['memberships']
            heatmap_data[i, :len(memberships)] = memberships
        
        # 绘制热力图
        im = axes[class_idx + 1].imshow(heatmap_data, aspect='auto',
                                       cmap='YlOrRd', vmin=0, vmax=1)
        
        axes[class_idx + 1].set_title(f'类别 {class_idx}', fontsize=11)
        axes[class_idx + 1].set_ylabel('Shapelet', fontsize=9)
        
        # 添加 colorbar
        plt.colorbar(im, ax=axes[class_idx + 1], label='隶属度 μ')
    
    axes[-1].set_xlabel('时间位置 t', fontsize=10)
    
    # 添加标题
    title = 'PseSC 隶属度热力图'
    if true_label is not None and predicted_label is not None:
        if true_label == predicted_label:
            title += f' | [正确] 预测: {predicted_label}, 真实: {true_label}'
        else:
            title += f' | [错误] 预测: {predicted_label}, 真实: {true_label}'
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] 热力图已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_membership_curves(memberships_dict: Dict,
                          class_idx: int = 0,
                          output_path: Optional[str] = None,
                          figsize: tuple = (12, 8)):
    """
    绘制单个类别所有 shapelet 的隶属度曲线
    
    Args:
        memberships_dict: 隶属度字典
        class_idx: 要绘制的类别索引
        output_path: 输出文件路径
        figsize: 图像大小
    """
    class_key = f'class_{class_idx}'
    class_memberships = memberships_dict[class_key]
    
    n_shapelets = len(class_memberships)
    n_cols = min(3, n_shapelets)
    n_rows = (n_shapelets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for i, shapelet_key in enumerate(sorted(class_memberships.keys())):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        shapelet_data = class_memberships[shapelet_key]
        memberships = shapelet_data['memberships']
        positions = shapelet_data['positions']
        
        # 绘制隶属度曲线
        ax.plot(positions, memberships, 'o-', linewidth=2, markersize=3)
        ax.fill_between(positions, 0, memberships, alpha=0.3)
        
        # 标记最大值
        max_pos = shapelet_data['max_pos']
        max_membership = shapelet_data['max_membership']
        ax.plot(max_pos, max_membership, 'r*', markersize=15, label='最大值')
        
        ax.set_title(f"{shapelet_key} (β={shapelet_data.get('beta', 'N/A'):.2f})",
                    fontsize=10)
        ax.set_xlabel('位置 t')
        ax.set_ylabel('隶属度 μ')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    
    # 隐藏多余的子图
    for i in range(n_shapelets, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'类别 {class_idx} 的隶属度曲线', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] 隶属度曲线已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_psesc_features(psesc_features: np.ndarray,
                       feature_names: Optional[List[str]] = None,
                       output_path: Optional[str] = None,
                       figsize: tuple = (14, 6)):
    """
    绘制 PseSC 特征分布
    
    Args:
        psesc_features: (n_samples, feature_dim) 或 (feature_dim,)
        feature_names: 特征名称列表
        output_path: 输出文件路径
        figsize: 图像大小
    """
    if psesc_features.ndim == 1:
        psesc_features = psesc_features.reshape(1, -1)
    
    n_samples, feature_dim = psesc_features.shape
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 左图：特征值分布（热力图）
    im = ax1.imshow(psesc_features.T, aspect='auto', cmap='viridis')
    ax1.set_xlabel('样本索引')
    ax1.set_ylabel('特征索引')
    ax1.set_title(f'PseSC 特征矩阵 ({n_samples} × {feature_dim})')
    plt.colorbar(im, ax=ax1, label='特征值')
    
    # 右图：特征统计
    feature_means = np.mean(psesc_features, axis=0)
    feature_stds = np.std(psesc_features, axis=0)
    
    x = np.arange(feature_dim)
    ax2.bar(x, feature_means, yerr=feature_stds, alpha=0.7, capsize=3)
    ax2.set_xlabel('特征索引')
    ax2.set_ylabel('特征值')
    ax2.set_title('特征均值 ± 标准差')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] PseSC 特征图已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_beta_distribution(adaptive_betas: Dict,
                          output_path: Optional[str] = None,
                          figsize: tuple = (10, 6)):
    """
    绘制自适应 β 值分布
    
    Args:
        adaptive_betas: {(class_idx, shapelet_idx): beta_value}
        output_path: 输出文件路径
        figsize: 图像大小
    """
    beta_values = list(adaptive_betas.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 直方图
    ax1.hist(beta_values, bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(np.median(beta_values), color='r', linestyle='--',
               label=f'中位数: {np.median(beta_values):.2f}')
    ax1.set_xlabel('β 值')
    ax1.set_ylabel('频数')
    ax1.set_title('自适应 β 值分布')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 按类别分组
    n_classes = max(k[0] for k in adaptive_betas.keys()) + 1
    class_betas = [[] for _ in range(n_classes)]
    
    for (class_idx, shapelet_idx), beta in adaptive_betas.items():
        class_betas[class_idx].append(beta)
    
    ax2.boxplot(class_betas, labels=[f'C{i}' for i in range(n_classes)])
    ax2.set_xlabel('类别')
    ax2.set_ylabel('β 值')
    ax2.set_title('各类别 β 值分布')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] β 分布图已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # 测试可视化
    print("可视化模块测试")
    print("=" * 50)
    
    # 模拟数据
    np.random.seed(42)
    
    # 模拟时间序列
    time_series = np.random.randn(6, 100)
    
    # 模拟隶属度字典
    memberships_dict = {}
    for class_idx in range(4):
        class_memberships = {}
        for shapelet_idx in range(3):
            T = 80
            memberships = np.random.rand(T) * 0.5
            positions = np.arange(T)
            
            class_memberships[f'shapelet_{shapelet_idx}'] = {
                'memberships': memberships,
                'positions': positions,
                'max_membership': np.max(memberships),
                'max_pos': np.argmax(memberships),
                'beta': 2.0 + np.random.randn() * 0.5
            }
        
        memberships_dict[f'class_{class_idx}'] = class_memberships
    
    # 测试热力图
    print("\n生成隶属度热力图...")
    plot_membership_heatmap(time_series, memberships_dict,
                          true_label=1, predicted_label=1,
                          output_path='test_heatmap.png')
    
    # 测试隶属度曲线
    print("\n生成隶属度曲线...")
    plot_membership_curves(memberships_dict, class_idx=0,
                         output_path='test_curves.png')
    
    print("\n✓ 可视化测试完成")
