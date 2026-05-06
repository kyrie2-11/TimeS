"""
PseSC 完整演示
=============
端到端 pipeline 示例：从 shapelet 池构建到分类
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 导入本地模块（同目录）
from shapelet_pool import build_shapelet_pool
from membership_mapping import MembershipMapper
from psesc_extractor import PseSCExtractor, diagnose_psesc_features
from visualization import plot_membership_heatmap, plot_membership_curves

# 尝试导入数据加载器
try:
    from Dataset.load_UEA_data import Data_Loader
except ImportError:
    print("[警告] 无法导入 Dataset.load_UEA_data.Data_Loader，将使用模拟数据")
    Data_Loader = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def demo_psesc_pipeline(data_name: str = 'BasicMotions',
                       n_per_class: int = 3,
                       beta_mode: str = 'adaptive',
                       visualize: bool = True):
    """
    完整 PseSC pipeline 演示
    
    Args:
        data_name: 数据集名称
        n_per_class: 每类 shapelet 数量
        beta_mode: 'adaptive' 或 'global'
        visualize: 是否生成可视化
    """
    print("=" * 70)
    print("PseSC 完整 Pipeline 演示")
    print("=" * 70)
    
    # ==================== 步骤 1: 加载数据 ====================
    print("\n[步骤 1] 加载数据...")
    config = {
        'data_dir': f'Dataset/UEA/Multivariate_ts/{data_name}',
        'pattern': 'TRAIN',
        'val_pattern': 'TEST'
    }
    
    try:
        if Data_Loader is None:
            raise ImportError("Data_Loader not available")
        Data = Data_Loader(config)
        train_data = Data['train_data']
        train_label = Data['train_label']
        test_data = Data['test_data']
        test_label = Data['test_label']
    except Exception as e:
        print(f"[错误] 数据加载失败: {e}")
        print("[提示] 使用模拟数据继续演示...")
        
        # 生成模拟数据
        np.random.seed(42)
        train_data = np.random.randn(32, 6, 100)
        train_label = np.random.randint(0, 4, 32)
        test_data = np.random.randn(40, 6, 100)
        test_label = np.random.randint(0, 4, 40)
    
    n_train, n_dims, seq_len = train_data.shape
    n_test = test_data.shape[0]
    n_classes = len(np.unique(train_label))
    
    print(f"✓ 数据加载完成")
    print(f"  - 训练集: {n_train} 样本")
    print(f"  - 测试集: {n_test} 样本")
    print(f"  - 维度: {n_dims}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 类别数: {n_classes}")
    
    # ==================== 步骤 2: 构建 Shapelet 池 ====================
    print(f"\n[步骤 2] 构建 Shapelet 池（每类 {n_per_class} 个）...")
    
    shapelets, shapelets_info = build_shapelet_pool(
        train_data, train_label,
        n_per_class=n_per_class,
        window_size=30,
        num_pip=0.2
    )
    
    total_shapelets = n_classes * n_per_class
    print(f"✓ Shapelet 池构建完成: {total_shapelets} 个")
    
    # ==================== 步骤 3: 隶属度映射 ====================
    print(f"\n[步骤 3] 配置隶属度映射器（模式: {beta_mode}）...")
    
    mapper = MembershipMapper(
        shapelets, shapelets_info,
        beta_mode=beta_mode,
        global_beta=2.0
    )
    
    if beta_mode == 'adaptive':
        mapper.calibrate_adaptive_beta(train_data)
        
        if visualize:
            print("  - β 分布统计...")
            betas = list(mapper.adaptive_betas.values())
            print(f"    平均 β: {np.mean(betas):.4f}")
            print(f"    β 范围: [{np.min(betas):.4f}, {np.max(betas):.4f}]")
    
    # ==================== 步骤 4: 提取 PseSC 特征 ====================
    print(f"\n[步骤 4] 提取 PseSC 特征...")
    
    extractor = PseSCExtractor(n_order_stats=3)
    
    # 训练集特征
    print("  - 处理训练集...")
    train_memberships_list = mapper.batch_compute_memberships(train_data)
    train_features = extractor.extract_batch_psesc(train_memberships_list)
    
    # 测试集特征
    print("  - 处理测试集...")
    test_memberships_list = mapper.batch_compute_memberships(test_data)
    test_features = extractor.extract_batch_psesc(test_memberships_list)
    
    print(f"✓ 特征提取完成")
    print(f"  - 训练集特征: {train_features.shape}")
    print(f"  - 测试集特征: {test_features.shape}")
    
    # 特征诊断
    print(f"\n[诊断] PseSC 特征质量...")
    feature_names = extractor.get_feature_names(n_classes, n_per_class)
    diag = diagnose_psesc_features(train_features, feature_names)
    
    print(f"  - 特征维度: {diag['feature_dim']}")
    print(f"  - 特征范围: [{np.min(train_features):.4f}, {np.max(train_features):.4f}]")
    print(f"  - 恒定特征数: {diag['n_constant_features']}")
    
    # ==================== 步骤 5: 分类 ====================
    print(f"\n[步骤 5] 训练分类器（Random Forest）...")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(train_features, train_label)
    
    # 预测
    train_pred = clf.predict(train_features)
    test_pred = clf.predict(test_features)
    
    # 评估
    train_acc = accuracy_score(train_label, train_pred)
    test_acc = accuracy_score(test_label, test_pred)
    
    print(f"✓ 分类完成")
    print(f"  - 训练集准确率: {train_acc * 100:.2f}%")
    print(f"  - 测试集准确率: {test_acc * 100:.2f}%")
    
    print(f"\n分类报告（测试集）:")
    print(classification_report(test_label, test_pred))
    
    print(f"\n混淆矩阵（测试集）:")
    print(confusion_matrix(test_label, test_pred))
    
    # ==================== 步骤 6: 可视化示例 ====================
    if visualize:
        print(f"\n[步骤 6] 生成可视化示例...")
        
        # 随机选择 3 个测试样本
        sample_indices = np.random.choice(n_test, min(3, n_test), replace=False)
        
        for idx in sample_indices:
            sample_ts = test_data[idx]
            sample_memberships = test_memberships_list[idx]
            true_label = test_label[idx]
            pred_label = test_pred[idx]
            
            # 热力图
            plot_membership_heatmap(
                sample_ts, sample_memberships,
                true_label=true_label,
                predicted_label=pred_label,
                output_path=f'sample_{idx}_heatmap.png'
            )
            
            # 隶属度曲线（选择预测的类别）
            plot_membership_curves(
                sample_memberships,
                class_idx=pred_label,
                output_path=f'sample_{idx}_curves.png'
            )
        
        print(f"✓ 可视化完成（生成 {len(sample_indices) * 2} 张图）")
    
    print("\n" + "=" * 70)
    print("PseSC Pipeline 演示完成！")
    print("=" * 70)
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_features': train_features,
        'test_features': test_features,
        'classifier': clf
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PseSC 完整演示')
    parser.add_argument('--data', type=str, default='BasicMotions',
                       help='数据集名称')
    parser.add_argument('--n_per_class', type=int, default=3,
                       help='每类 shapelet 数量')
    parser.add_argument('--beta_mode', type=str, default='adaptive',
                       choices=['adaptive', 'global'],
                       help='β 模式')
    parser.add_argument('--no_viz', action='store_true',
                       help='禁用可视化')
    
    args = parser.parse_args()
    
    results = demo_psesc_pipeline(
        data_name=args.data,
        n_per_class=args.n_per_class,
        beta_mode=args.beta_mode,
        visualize=not args.no_viz
    )
    
    print(f"\n最终结果:")
    print(f"  训练准确率: {results['train_acc'] * 100:.2f}%")
    print(f"  测试准确率: {results['test_acc'] * 100:.2f}%")
