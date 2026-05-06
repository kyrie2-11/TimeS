"""
ECG 心电图分类实验
使用 PseSC 特征 + 决策树/随机森林进行时间序列分类
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.decision_tree_classifier import PseSCDecisionTreeClassifier
from models.data_loader import load_ucr_dataset, convert_to_list


def run_classification_experiment(dataset_name='ECGFiveDays'):
    """
    运行完整的分类实验
    
    参数：
        dataset_name: 数据集名称
    """
    print("\n" + "="*70)
    print(f"  ECG 心电图分类实验 - {dataset_name} 数据集")
    print("  基于 PseSC 特征 + 决策树模型")
    print("="*70)
    
    # ============ 1. 加载数据 ============
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'Datasets', dataset_name)
    train_path = os.path.join(data_dir, f'{dataset_name}_TRAIN')
    test_path = os.path.join(data_dir, f'{dataset_name}_TEST')
    
    X_train, y_train, X_test, y_test = load_ucr_dataset(train_path, test_path)
    
    # 转换为列表格式（适配 shapelet_pool 接口）
    X_train_list = convert_to_list(X_train)
    X_test_list = convert_to_list(X_test)
    
    # 创建输出目录
    output_dir = os.path.join(project_root, 'results', 'classification')
    os.makedirs(output_dir, exist_ok=True)
    
    # ============ 2. 实验 A：单棵决策树 ============
    print("\n" + "#"*70)
    print("  实验 A：单棵决策树 + PseSC")
    print("#"*70)
    
    dt_classifier = PseSCDecisionTreeClassifier(
        n_shapelets_per_class=3,
        beta_mode='adaptive',
        tree_params={
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        },
        use_random_forest=False
    )
    
    # 训练
    dt_classifier.fit(X_train_list, y_train)
    
    # 评估
    metrics_dt = dt_classifier.evaluate(X_test_list, y_test, 
                                        plot_cm=True, 
                                        save_path=output_dir)
    
    print(f"\n【结果】")
    print(f"  准确率 (Accuracy): {metrics_dt['accuracy']:.4f}")
    print(f"  F1 分数 (F1-Score): {metrics_dt['f1_score']:.4f}")
    
    # 特征重要性分析
    print("\n分析特征重要性...")
    importance_df_dt = dt_classifier.get_feature_importance(
        top_k=15, 
        save_path=output_dir
    )
    
    # 可视化决策树
    print("\n可视化决策树结构...")
    dt_classifier.visualize_tree(max_depth=3, save_path=output_dir)
    
    # ============ 3. 实验 B：随机森林 ============
    print("\n" + "#"*70)
    print("  实验 B：随机森林 + PseSC")
    print("#"*70)
    
    rf_classifier = PseSCDecisionTreeClassifier(
        n_shapelets_per_class=5,  # 使用更多 shapelet
        beta_mode='adaptive',
        tree_params={
            'max_depth': 12,
            'n_estimators': 100,
            'min_samples_split': 5
        },
        use_random_forest=True
    )
    
    # 训练
    rf_classifier.fit(X_train_list, y_train)
    
    # 评估
    metrics_rf = rf_classifier.evaluate(X_test_list, y_test, 
                                        plot_cm=True, 
                                        save_path=output_dir)
    
    print(f"\n【结果】")
    print(f"  准确率 (Accuracy): {metrics_rf['accuracy']:.4f}")
    print(f"  F1 分数 (F1-Score): {metrics_rf['f1_score']:.4f}")
    
    # 特征重要性分析
    print("\n分析特征重要性...")
    importance_df_rf = rf_classifier.get_feature_importance(
        top_k=15, 
        save_path=output_dir
    )
    
    # ============ 4. 结果对比与可视化 ============
    print("\n" + "="*70)
    print("  最终结果对比")
    print("="*70)
    
    results_df = pd.DataFrame({
        '模型': ['决策树 (Decision Tree)', '随机森林 (Random Forest)'],
        '准确率': [metrics_dt['accuracy'], metrics_rf['accuracy']],
        'F1 分数': [metrics_dt['f1_score'], metrics_rf['f1_score']]
    })
    
    print("\n" + results_df.to_string(index=False))
    
    # 保存结果到 CSV
    results_path = f'{output_dir}/classification_results.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存到 {results_path}")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = ['Decision Tree', 'Random Forest']
    accuracies = [metrics_dt['accuracy'], metrics_rf['accuracy']]
    f1_scores = [metrics_dt['f1_score'], metrics_rf['f1_score']]
    
    # 准确率对比
    bars1 = axes[0].bar(models, accuracies, color=['#3498db', '#2ecc71'], 
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # F1 分数对比
    bars2 = axes[1].bar(models, f1_scores, color=['#3498db', '#2ecc71'], 
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    comparison_path = f'{output_dir}/model_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ 模型对比图已保存为 {comparison_path}")
    plt.close()
    
    # ============ 5. 总结 ============
    print("\n" + "="*70)
    print("  实验总结")
    print("="*70)
    print(f"✓ 所有结果已保存到目录: {output_dir}")
    print(f"✓ 生成文件:")
    print(f"  - confusion_matrix_decisiontree.png (决策树混淆矩阵)")
    print(f"  - confusion_matrix_randomforest.png (随机森林混淆矩阵)")
    print(f"  - feature_importance_decisiontree.png (决策树特征重要性)")
    print(f"  - feature_importance_randomforest.png (随机森林特征重要性)")
    print(f"  - decision_tree_viz.png (决策树结构可视化)")
    print(f"  - model_comparison.png (模型性能对比)")
    print(f"  - classification_results.csv (结果汇总)")
    
    # 返回最佳模型
    if metrics_rf['accuracy'] > metrics_dt['accuracy']:
        print(f"\n🏆 最佳模型: 随机森林 (准确率: {metrics_rf['accuracy']:.4f})")
        return rf_classifier, metrics_rf
    else:
        print(f"\n🏆 最佳模型: 决策树 (准确率: {metrics_dt['accuracy']:.4f})")
        return dt_classifier, metrics_dt


if __name__ == '__main__':
    try:
        best_model, best_metrics = run_classification_experiment('ECGFiveDays')
        print("\n✓ 实验成功完成！")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
