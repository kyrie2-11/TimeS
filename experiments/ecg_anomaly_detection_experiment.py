"""
ECG 心电图异常检测实验
使用 PseSC 特征 + Isolation Forest / One-Class SVM
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.anomaly_detector import PseSCAnomalyDetector
from models.data_loader import load_ecg_for_anomaly_detection, convert_to_list


def run_anomaly_detection_experiment(dataset_name='ECGFiveDays', normal_label=1):
    """
    运行完整的异常检测实验
    
    参数：
        dataset_name: 数据集名称
        normal_label: 哪个类别标记为正常
    """
    print("\n" + "="*70)
    print(f"  ECG 心电图异常检测实验 - {dataset_name} 数据集")
    print("  基于 PseSC 特征 + 异常检测算法")
    print("="*70)
    
    # ============ 1. 加载数据 ============
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'Datasets', dataset_name)
    train_path = os.path.join(data_dir, f'{dataset_name}_TRAIN')
    test_path = os.path.join(data_dir, f'{dataset_name}_TEST')
    
    X_normal, X_test, y_test = load_ecg_for_anomaly_detection(
        train_path, test_path, normal_label=normal_label
    )
    
    # 转换为列表格式
    X_normal_list = convert_to_list(X_normal)
    X_test_list = convert_to_list(X_test)
    
    # 创建输出目录
    output_dir = os.path.join(project_root, 'results', 'anomaly_detection')
    os.makedirs(output_dir, exist_ok=True)
    
    # ============ 2. 方法1：Isolation Forest ============
    print("\n" + "#"*70)
    print("  方法 1: Isolation Forest")
    print("#"*70)
    
    detector_iforest = PseSCAnomalyDetector(
        n_shapelets_per_class=5,
        beta_mode='adaptive',
        contamination=0.1,
        method='iforest'
    )
    
    # 训练（仅使用正常样本）
    detector_iforest.fit(X_normal_list)
    
    # 评估
    metrics_iforest = detector_iforest.evaluate(
        X_test_list, y_test, 
        save_path=output_dir
    )
    
    print(f"\n【Isolation Forest 模型结果】")
    print(f"  准确率 (Accuracy): {metrics_iforest['accuracy']:.4f}")
    print(f"  AUC-ROC: {metrics_iforest['auc']:.4f}")
    
    # ============ 3. 方法2：One-Class SVM ============
    print("\n" + "#"*70)
    print("  方法 2: One-Class SVM")
    print("#"*70)
    
    detector_svm = PseSCAnomalyDetector(
        n_shapelets_per_class=5,
        beta_mode='adaptive',
        contamination=0.1,
        method='svm'
    )
    
    # 训练
    detector_svm.fit(X_normal_list)
    
    # 评估
    metrics_svm = detector_svm.evaluate(
        X_test_list, y_test, 
        save_path=output_dir
    )
    
    print(f"\n【SVM 模型结果】")
    print(f"  准确率 (Accuracy): {metrics_svm['accuracy']:.4f}")
    print(f"  AUC-ROC: {metrics_svm['auc']:.4f}")
    
    # ============ 4. 异常定位示例 ============
    print("\n" + "#"*70)
    print("  异常定位与可解释性分析")
    print("#"*70)
    
    # 找到一个异常样本进行分析
    anomaly_indices = np.where(y_test == 0)[0]
    if len(anomaly_indices) > 0:
        sample_idx = anomaly_indices[0]
        sample_ts = X_test_list[sample_idx]
        
        print(f"\n分析异常样本 #{sample_idx}...")
        
        # 定位异常
        anomaly_info = detector_iforest.locate_anomaly(sample_ts, threshold_percentile=95)
        
        print(f"检测到 {len(anomaly_info['anomaly_positions'])} 个可疑位置:")
        for pos_info in anomaly_info['anomaly_positions'][:5]:  # 显示前5个
            print(f"  - 时间位置 {pos_info['time_position']}, "
                  f"异常强度 {pos_info['anomaly_intensity']:.3f}")
        
        # 可视化异常热力图
        heatmap_path = f'{output_dir}/anomaly_heatmap_sample_{sample_idx}.png'
        detector_iforest.visualize_anomaly_heatmap(sample_ts, save_path=heatmap_path)
        
        # 同时分析一个正常样本作为对比
        normal_indices = np.where(y_test == 1)[0]
        if len(normal_indices) > 0:
            normal_idx = normal_indices[0]
            normal_ts = X_test_list[normal_idx]
            heatmap_normal_path = f'{output_dir}/normal_heatmap_sample_{normal_idx}.png'
            detector_iforest.visualize_anomaly_heatmap(normal_ts, save_path=heatmap_normal_path)
            print(f"\n✓ 正常样本热力图已保存（对比用）")
    
    # ============ 5. 结果对比 ============
    print("\n" + "="*70)
    print("  异常检测方法对比")
    print("="*70)
    
    results_df = pd.DataFrame({
        '方法': ['Isolation Forest', 'One-Class SVM'],
        '准确率': [metrics_iforest['accuracy'], metrics_svm['accuracy']],
        'AUC-ROC': [metrics_iforest['auc'], metrics_svm['auc']]
    })
    
    print("\n" + results_df.to_string(index=False))
    
    # 保存结果
    results_path = f'{output_dir}/anomaly_detection_results.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存到 {results_path}")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['Isolation Forest', 'One-Class SVM']
    accuracies = [metrics_iforest['accuracy'], metrics_svm['accuracy']]
    aucs = [metrics_iforest['auc'], metrics_svm['auc']]
    
    # 准确率对比
    bars1 = axes[0].bar(methods, accuracies, color=['#e74c3c', '#9b59b6'], 
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # AUC 对比
    bars2 = axes[1].bar(methods, aucs, color=['#e74c3c', '#9b59b6'], 
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    axes[1].set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    comparison_path = f'{output_dir}/method_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ 方法对比图已保存为 {comparison_path}")
    plt.close()
    
    # ============ 6. 总结 ============
    print("\n" + "="*70)
    print("  实验总结")
    print("="*70)
    print(f"✓ 所有结果已保存到目录: {output_dir}")
    print(f"✓ 生成文件:")
    print(f"  - anomaly_curves_iforest.png (Isolation Forest ROC/PR 曲线)")
    print(f"  - anomaly_curves_svm.png (SVM ROC/PR 曲线)")
    print(f"  - anomaly_heatmap_sample_*.png (异常样本热力图)")
    print(f"  - normal_heatmap_sample_*.png (正常样本热力图)")
    print(f"  - method_comparison.png (方法性能对比)")
    print(f"  - anomaly_detection_results.csv (结果汇总)")
    
    # 返回最佳模型
    if metrics_iforest['auc'] > metrics_svm['auc']:
        print(f"\n🏆 最佳方法: Isolation Forest (AUC: {metrics_iforest['auc']:.4f})")
        return detector_iforest, metrics_iforest
    else:
        print(f"\n🏆 最佳方法: SVM (AUC: {metrics_svm['auc']:.4f})")
        return detector_svm, metrics_svm


if __name__ == '__main__':
    try:
        best_detector, best_metrics = run_anomaly_detection_experiment(
            dataset_name='ECGFiveDays',
            normal_label=1
        )
        print("\n✓ 实验成功完成！")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
