#!/usr/bin/env python
"""
超快速测试脚本：5分钟内完成验证
使用小样本和少量Shapelet进行快速功能测试
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.data_loader import load_smap_msl_single_channel, get_smap_msl_channel_list
from models.multivariate_anomaly_detector import MultiVariateAnomalyDetector


def ultra_fast_test():
    """超快速测试（仅用于验证功能）"""
    
    print("\n" + "="*70)
    print("[超快速测试] 多变量异常检测系统 (5分钟验证)")
    print("="*70)
    
    # 第1步：获取通道
    print("\n[1] 获取可用通道...")
    channels = get_smap_msl_channel_list()
    smap_channels = channels.get('SMAP', [])
    if not smap_channels:
        print("✗ 错误：未找到SMAP通道")
        return False
    
    test_channel = smap_channels[0]
    print(f"✓ 使用通道: {test_channel}")
    
    # 第2步：加载数据
    print(f"\n[2] 加载数据...")
    try:
        X_train_full, X_test_full, y_test = load_smap_msl_single_channel(test_channel)
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False
    
    # 使用数据子集进行快速测试
    seq_len_train, n_dims = X_train_full.shape
    
    # 取前500个样本进行训练（快速）
    X_train = X_train_full[:min(500, seq_len_train), :]
    # 取前500个样本进行测试（快速）
    X_test = X_test_full[:min(500, X_test_full.shape[0]), :]
    y_test_subset = y_test[:min(500, len(y_test))]
    
    print(f"✓ 数据准备完成 (使用子集快速测试)")
    print(f"  训练: {X_train.shape}, 测试: {X_test.shape}")
    print(f"  维度: {n_dims}, 异常比例: {(1-y_test_subset.mean())*100:.1f}%")
    
    # 第3步：初始化和训练
    print(f"\n[3] 初始化检测器...")
    try:
        # 使用较少的Shapelet以加快速度
        detector = MultiVariateAnomalyDetector(
            window_sizes=[15, 25],  # 减少尺度数量
            n_shapelets_per_scale=5,  # 大幅减少Shapelet数
            detection_method='iforest',
            contamination=0.1,
            use_periodicity=False,  # 跳过周期检测以加快
            use_temporal_window=False  # 跳过时间窗口以加快
        )
        print(f"✓ 检测器初始化完成")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    print(f"\n[4] 训练模型（可能需要1-2分钟）...")
    try:
        detector.fit(X_train)
        print(f"✓ 模型训练完成")
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 第4步：预测
    print(f"\n[5] 异常检测...")
    try:
        predictions, anomaly_scores = detector.predict(X_test)
        print(f"✓ 检测完成")
        print(f"  异常数: {predictions.sum()}/{len(predictions)}")
        print(f"  分数范围: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 第5步：快速评估
    print(f"\n[6] 性能评估...")
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        
        # 对齐长度
        if len(predictions) < len(y_test_subset):
            padded_pred = np.concatenate([
                predictions,
                np.full(len(y_test_subset) - len(predictions), predictions[-1])
            ])
            padded_scores = np.concatenate([
                anomaly_scores,
                np.full(len(y_test_subset) - len(anomaly_scores), anomaly_scores[-1])
            ])
        else:
            padded_pred = predictions[:len(y_test_subset)]
            padded_scores = anomaly_scores[:len(y_test_subset)]
        
        auc = roc_auc_score(y_test_subset, padded_scores)
        f1 = f1_score(y_test_subset, padded_pred)
        
        print(f"✓ 评估完成")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    except Exception as e:
        print(f"✗ 评估失败: {e}")
        return False
    
    print(f"\n{'='*70}")
    print(f"✓ 超快速测试成功！系统完全可用")
    print(f"{'='*70}\n")
    print(f"下一步：运行完整实验")
    print(f"  python experiments/multivariate_anomaly_experiment.py --mode benchmark --n-channels 3\n")
    
    return True


if __name__ == '__main__':
    success = ultra_fast_test()
    sys.exit(0 if success else 1)
