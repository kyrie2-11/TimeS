#!/usr/bin/env python
"""
快速测试脚本：验证系统端到端工作流
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.data_loader import load_smap_msl_single_channel, get_smap_msl_channel_list
from models.multivariate_anomaly_detector import MultiVariateAnomalyDetector


def quick_test():
    """快速测试单个SMAP通道"""
    
    print("\n" + "="*70)
    print("[快速测试] 多变量异常检测系统")
    print("="*70)
    
    # 获取可用通道
    print("\n[第1步] 获取可用通道...")
    channels = get_smap_msl_channel_list()
    
    smap_channels = channels.get('SMAP', [])
    if not smap_channels:
        print("✗ 错误：未找到SMAP通道")
        return False
    
    test_channel = smap_channels[0]
    print(f"✓ 使用通道: {test_channel}")
    print(f"  SMAP共有: {len(channels.get('SMAP', []))} 个通道")
    print(f"  MSL共有: {len(channels.get('MSL', []))} 个通道")
    
    # 加载数据
    print(f"\n[第2步] 加载数据: {test_channel}...")
    try:
        X_train, X_test, y_test = load_smap_msl_single_channel(test_channel)
        print(f"✓ 数据加载完成")
        print(f"  训练集: {X_train.shape}")
        print(f"  测试集: {X_test.shape}")
        print(f"  异常样本: {(1-y_test).sum()}/{len(y_test)}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    
    # 初始化检测器
    print(f"\n[第3步] 初始化检测器...")
    try:
        seq_len_train, n_dims = X_train.shape
        
        # 调整window_sizes以适应数据长度
        window_sizes = [w for w in [10, 20, 30, 40] if w < seq_len_train]
        if not window_sizes:
            window_sizes = [max(5, seq_len_train // 4)]
        
        detector = MultiVariateAnomalyDetector(
            window_sizes=window_sizes,
            n_shapelets_per_scale=10,  # 快速测试用较少数量
            detection_method='iforest',
            contamination=0.1,
            use_periodicity=True,
            use_temporal_window=True,
            temporal_window_size=3
        )
        print(f"✓ 检测器初始化完成")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 训练
    print(f"\n[第4步] 训练模型...")
    try:
        detector.fit(X_train)
        print(f"✓ 模型训练完成")
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 预测
    print(f"\n[第5步] 异常检测...")
    try:
        predictions, anomaly_scores = detector.predict(X_test)
        print(f"✓ 检测完成")
        print(f"  预测长度: {len(predictions)}")
        print(f"  异常检测数: {predictions.sum()}")
        print(f"  异常分数范围: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 评估
    print(f"\n[第6步] 性能评估...")
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        
        # 对齐长度
        if len(predictions) < len(y_test):
            padded_pred = np.concatenate([
                predictions,
                np.full(len(y_test) - len(predictions), predictions[-1])
            ])
            padded_scores = np.concatenate([
                anomaly_scores,
                np.full(len(y_test) - len(anomaly_scores), anomaly_scores[-1])
            ])
        else:
            padded_pred = predictions[:len(y_test)]
            padded_scores = anomaly_scores[:len(y_test)]
        
        auc = roc_auc_score(y_test, padded_scores)
        f1 = f1_score(y_test, padded_pred)
        
        print(f"✓ 评估完成")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    except Exception as e:
        print(f"✗ 评估失败: {e}")
        return False
    
    print(f"\n{'='*70}")
    print(f"✓ 快速测试通过！系统工作正常")
    print(f"{'='*70}\n")
    
    return True


if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)
