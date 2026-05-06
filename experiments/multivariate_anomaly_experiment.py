"""
多变量异常检测实验脚本
=====================
在SMAP/MSL数据集上评估改进的异常检测系统
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_loader import load_smap_msl_single_channel, get_smap_msl_channel_list
from models.multivariate_anomaly_detector import MultiVariateAnomalyDetector
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix


def evaluate_single_channel(channel_name: str, 
                          window_sizes: list = None,
                          detection_method: str = 'iforest',
                          use_periodicity: bool = True,
                          n_shapelets: int = 20) -> Dict:
    """
    评估单个通道的异常检测性能
    
    Args:
        channel_name: 通道名称如'P-1'
        window_sizes: 时间尺度
        detection_method: 检测方法
        use_periodicity: 是否使用周期检测
        n_shapelets: 每个尺度的Shapelet数量
    
    Returns:
        results: 性能指标字典
    """
    if window_sizes is None:
        window_sizes = [10, 20, 30, 40]
    
    print(f"\n{'='*70}")
    print(f"[实验] 评估通道: {channel_name}")
    print(f"{'='*70}")
    
    try:
        # 加载数据
        print(f"[数据加载] 正在加载 {channel_name}...")
        X_train, X_test, y_test = load_smap_msl_single_channel(channel_name)
        
        seq_len_train, n_dims = X_train.shape
        seq_len_test = X_test.shape[0]
        n_anomaly = (1 - y_test).sum()
        
        print(f"  训练数据: {X_train.shape}")
        print(f"  测试数据: {X_test.shape}")
        print(f"  异常数: {int(n_anomaly)} / {len(y_test)} ({n_anomaly/len(y_test)*100:.1f}%)")
        
        # 调整window_sizes以适应数据长度
        adjusted_window_sizes = [w for w in window_sizes if w < seq_len_train]
        if len(adjusted_window_sizes) == 0:
            adjusted_window_sizes = [max(5, seq_len_train // 4)]
        
        print(f"  调整后的窗口大小: {adjusted_window_sizes}")
        
        # 初始化检测器
        print(f"\n[模型初始化]")
        detector = MultiVariateAnomalyDetector(
            window_sizes=adjusted_window_sizes,
            n_shapelets_per_scale=min(n_shapelets, seq_len_train // 10),
            detection_method=detection_method,
            contamination=0.1,
            use_periodicity=use_periodicity,
            use_temporal_window=True,
            temporal_window_size=5
        )
        
        # 训练
        print(f"\n[模型训练]")
        detector.fit(X_train)
        
        # 预测
        print(f"\n[模型预测]")
        predictions, anomaly_scores = detector.predict(X_test)
        
        # 评估性能
        print(f"\n[性能评估]")
        
        # 对齐预测和标签（由于窗口，预测序列可能更短）
        if len(predictions) < len(y_test):
            # 填充到相同长度（用最后的值）
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
        
        # 计算指标
        auc_roc = roc_auc_score(y_test, padded_scores)
        f1 = f1_score(y_test, padded_pred)
        precision = precision_score(y_test, padded_pred, zero_division=0)
        recall = recall_score(y_test, padded_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_test, padded_pred, labels=[0, 1]).ravel()
        
        print(f"  AUC-ROC:  {auc_roc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:   {recall:.4f}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        
        results = {
            'channel': channel_name,
            'window_sizes': adjusted_window_sizes,
            'n_dims': n_dims,
            'seq_len_train': seq_len_train,
            'seq_len_test': seq_len_test,
            'n_anomaly': int(n_anomaly),
            'auc_roc': auc_roc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'status': 'success'
        }
        
    except Exception as e:
        print(f"  ✗ 错误: {str(e)}")
        results = {
            'channel': channel_name,
            'status': 'error',
            'error_msg': str(e)
        }
    
    return results


def run_smap_msl_benchmark(n_channels_per_dataset: int = 5,
                          window_sizes: list = None,
                          detection_method: str = 'iforest',
                          use_periodicity: bool = True) -> pd.DataFrame:
    """
    在SMAP/MSL数据集上运行基准测试
    
    Args:
        n_channels_per_dataset: 每个数据集(SMAP/MSL)的测试通道数
        window_sizes: 时间尺度
        detection_method: 检测方法
        use_periodicity: 是否使用周期检测
    
    Returns:
        results_df: 结果DataFrame
    """
    if window_sizes is None:
        window_sizes = [10, 20, 30, 40]
    
    print(f"\n{'='*70}")
    print(f"SMAP/MSL多变量异常检测基准测试")
    print(f"{'='*70}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置:")
    print(f"  - 时间尺度: {window_sizes}")
    print(f"  - 检测方法: {detection_method}")
    print(f"  - 周期检测: {use_periodicity}")
    print(f"  - 每个数据集的通道数: {n_channels_per_dataset}")
    
    # 获取可用通道
    channels = get_smap_msl_channel_list()
    
    print(f"\n可用通道数:")
    for dataset, chans in channels.items():
        print(f"  {dataset}: {len(chans)} 个")
    
    # 选择要测试的通道
    test_channels = []
    for dataset in ['SMAP', 'MSL']:
        if dataset in channels:
            selected = channels[dataset][:min(n_channels_per_dataset, len(channels[dataset]))]
            test_channels.extend(selected)
    
    print(f"\n选定测试通道: {len(test_channels)} 个")
    for i, ch in enumerate(test_channels, 1):
        print(f"  {i}. {ch}")
    
    # 评估每个通道
    all_results = []
    successful = 0
    failed = 0
    
    for i, channel in enumerate(test_channels, 1):
        print(f"\n[进度 {i}/{len(test_channels)}]")
        results = evaluate_single_channel(
            channel,
            window_sizes=window_sizes,
            detection_method=detection_method,
            use_periodicity=use_periodicity,
            n_shapelets=20
        )
        all_results.append(results)
        
        if results['status'] == 'success':
            successful += 1
        else:
            failed += 1
    
    # 生成结果表
    results_df = pd.DataFrame(all_results)
    
    print(f"\n{'='*70}")
    print(f"实验完成统计")
    print(f"{'='*70}")
    print(f"成功: {successful}, 失败: {failed}")
    
    if successful > 0:
        success_results = results_df[results_df['status'] == 'success']
        
        print(f"\n平均性能:")
        print(f"  AUC-ROC:  {success_results['auc_roc'].mean():.4f} ± {success_results['auc_roc'].std():.4f}")
        print(f"  F1-Score: {success_results['f1_score'].mean():.4f} ± {success_results['f1_score'].std():.4f}")
        print(f"  Precision: {success_results['precision'].mean():.4f} ± {success_results['precision'].std():.4f}")
        print(f"  Recall:   {success_results['recall'].mean():.4f} ± {success_results['recall'].std():.4f}")
        
        print(f"\n详细结果:")
        print(success_results[['channel', 'n_dims', 'auc_roc', 'f1_score', 'precision', 'recall']])
    
    # 保存结果
    output_dir = Path(__file__).parent.parent / "results" / "multivariate_anomaly"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✓ 结果已保存到: {output_file}")
    
    return results_df


def ablation_study(channel_name: str = 'P-1') -> Dict:
    """
    消融实验：验证各个创新点的贡献
    
    测试配置:
    1. 仅单维Shapelet (无多维)
    2. 无多尺度聚合 (仅单一尺度)
    3. 无周期检测
    4. 完整模型
    """
    print(f"\n{'='*70}")
    print(f"[消融实验] 通道: {channel_name}")
    print(f"{'='*70}")
    
    # 加载数据
    X_train, X_test, y_test = load_smap_msl_single_channel(channel_name)
    
    configurations = [
        {
            'name': '仅单维Shapelet',
            'window_sizes': [30],
            'use_periodicity': False,
        },
        {
            'name': '无多尺度聚合',
            'window_sizes': [30],
            'use_periodicity': True,
        },
        {
            'name': '无周期检测',
            'window_sizes': [10, 20, 30, 40],
            'use_periodicity': False,
        },
        {
            'name': '完整模型（三项创新）',
            'window_sizes': [10, 20, 30, 40],
            'use_periodicity': True,
        },
    ]
    
    ablation_results = {}
    
    for config in configurations:
        print(f"\n测试: {config['name']}")
        print(f"  - 窗口大小: {config['window_sizes']}")
        print(f"  - 周期检测: {config['use_periodicity']}")
        
        try:
            detector = MultiVariateAnomalyDetector(
                window_sizes=config['window_sizes'],
                n_shapelets_per_scale=20,
                detection_method='iforest',
                contamination=0.1,
                use_periodicity=config['use_periodicity'],
                use_temporal_window=True,
                temporal_window_size=5
            )
            
            detector.fit(X_train)
            predictions, anomaly_scores = detector.predict(X_test)
            
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
            
            ablation_results[config['name']] = {'auc': auc, 'f1': f1}
            
            print(f"  ✓ AUC-ROC: {auc:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            print(f"  ✗ 失败: {str(e)}")
            ablation_results[config['name']] = {'auc': 0, 'f1': 0}
    
    # 输出消融结果对比
    print(f"\n{'='*70}")
    print(f"消融实验结果汇总:")
    print(f"{'='*70}")
    
    full_auc = ablation_results.get('完整模型（三项创新）', {}).get('auc', 0)
    
    for config_name, metrics in ablation_results.items():
        if config_name != '完整模型（三项创新）':
            auc_drop = (full_auc - metrics['auc']) * 100
            print(f"{config_name}:")
            print(f"  AUC: {metrics['auc']:.4f} (下降 {auc_drop:.2f}%)")
            print(f"  F1:  {metrics['f1']:.4f}")
        else:
            print(f"{config_name}:")
            print(f"  AUC: {metrics['auc']:.4f} (基准)")
            print(f"  F1:  {metrics['f1']:.4f} (基准)")
    
    return ablation_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='多变量异常检测实验')
    parser.add_argument('--mode', choices=['benchmark', 'ablation', 'single'],
                       default='benchmark', help='实验模式')
    parser.add_argument('--channel', default='P-1', help='通道名称（single模式下使用）')
    parser.add_argument('--n-channels', type=int, default=5, help='每个数据集的通道数')
    parser.add_argument('--detection-method', default='iforest', 
                       choices=['iforest', 'svm', 'elliptic'], help='检测方法')
    parser.add_argument('--use-periodicity', action='store_true', default=True,
                       help='是否使用周期检测')
    
    args = parser.parse_args()
    
    if args.mode == 'benchmark':
        results = run_smap_msl_benchmark(
            n_channels_per_dataset=args.n_channels,
            detection_method=args.detection_method,
            use_periodicity=args.use_periodicity
        )
    
    elif args.mode == 'ablation':
        ablation_results = ablation_study(channel_name=args.channel)
    
    else:  # single
        result = evaluate_single_channel(
            args.channel,
            detection_method=args.detection_method,
            use_periodicity=args.use_periodicity
        )
        print(f"\n结果:")
        print(json.dumps(result, indent=2))
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
