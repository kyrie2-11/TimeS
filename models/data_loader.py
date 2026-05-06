"""
ECG 数据集加载与预处理工具
支持 UCR 时间序列数据集格式
"""

import numpy as np
import os


def load_ucr_dataset(train_path, test_path):
    """
    加载 UCR 格式的时间序列数据集
    
    UCR 格式：第一列是标签，其余列是时间序列值
    
    参数：
        train_path: 训练集文件路径
        test_path: 测试集文件路径
    
    返回：
        X_train: 训练时间序列 [N_train, L]
        y_train: 训练标签 [N_train]
        X_test: 测试时间序列 [N_test, L]
        y_test: 测试标签 [N_test]
    """
    # 加载训练集
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集文件未找到: {train_path}")
    
    train_data = np.loadtxt(train_path, delimiter=',')
    y_train = train_data[:, 0].astype(int)
    X_train = train_data[:, 1:]
    
    # 加载测试集
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试集文件未找到: {test_path}")
    
    test_data = np.loadtxt(test_path, delimiter=',')
    y_test = test_data[:, 0].astype(int)
    X_test = test_data[:, 1:]
    
    print("\n" + "="*60)
    print("数据集加载完成")
    print("="*60)
    print(f"训练集: {X_train.shape[0]} 个样本, 序列长度: {X_train.shape[1]}")
    print(f"测试集: {X_test.shape[0]} 个样本, 序列长度: {X_test.shape[1]}")
    print(f"类别数: {len(np.unique(y_train))}")
    print(f"训练集类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"测试集类别分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    return X_train, y_train, X_test, y_test


def load_ecg_for_anomaly_detection(train_path, test_path, normal_label=1):
    """
    为异常检测任务加载 ECG 数据
    
    将一个类别标记为"正常"，其他类别标记为"异常"
    
    参数：
        train_path: 训练集路径
        test_path: 测试集路径
        normal_label: 哪个标签代表正常样本
    
    返回：
        X_normal: 正常样本（用于训练）
        X_test: 测试集
        y_test: 测试标签 (1=正常, 0=异常)
    """
    # 加载完整数据
    train_data = np.loadtxt(train_path, delimiter=',')
    test_data = np.loadtxt(test_path, delimiter=',')
    
    # 提取正常样本（仅用于训练）
    normal_mask = train_data[:, 0] == normal_label
    X_normal = train_data[normal_mask, 1:]
    
    # 测试集
    X_test = test_data[:, 1:]
    # 将标签转换为二分类：正常=1, 异常=0
    y_test = (test_data[:, 0] == normal_label).astype(int)
    
    print("\n" + "="*60)
    print("异常检测数据集加载完成")
    print("="*60)
    print(f"正常样本（训练用）: {X_normal.shape[0]} 个")
    print(f"测试集: 正常={y_test.sum()}, 异常={len(y_test)-y_test.sum()}")
    print(f"序列长度: {X_test.shape[1]}")
    
    return X_normal, X_test, y_test


def convert_to_list(X):
    """
    将 numpy 数组转换为列表（适配某些接口）
    
    参数：
        X: [N, L] numpy 数组
    
    返回：
        list of arrays
    """
    return [X[i] for i in range(len(X))]


def z_normalize(X):
    """
    Z-score 标准化时间序列
    
    参数：
        X: [N, L] 或 [L] 时间序列
    
    返回：
        标准化后的时间序列
    """
    if X.ndim == 1:
        mean = np.mean(X)
        std = np.std(X)
        if std < 1e-8:
            return np.zeros_like(X)
        return (X - mean) / std
    else:
        normalized = np.zeros_like(X)
        for i in range(len(X)):
            normalized[i] = z_normalize(X[i])
        return normalized


import json
from pathlib import Path
import csv as csv_module


def load_smap_msl_single_channel(channel_name: str, data_dir: str = "Datasets/SMAP&MSL",
                                  test_label_file: str = "labeled_anomalies.csv"):
    """
    加载单个SMAP/MSL通道的多变量时间序列数据
    
    参数：
        channel_name: 通道名称，如'P-1', 'E-1', 'M-1', 'L-1'等
        data_dir: 数据集所在目录
        test_label_file: 包含异常标注的CSV文件
    
    返回：
        X_train: [seq_len, n_dims] - 训练集（正常）
        X_test: [seq_len, n_dims] - 测试集
        y_test: [seq_len] - 测试集标注 (1=正常, 0=异常)
    """
    train_path = os.path.join(data_dir, "train", f"{channel_name}.npy")
    test_path = os.path.join(data_dir, "test", f"{channel_name}.npy")
    label_path = os.path.join(data_dir, test_label_file)
    
    # 加载数据
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到通道 {channel_name} 的数据文件")
    
    X_train = np.load(train_path)  # [seq_len, n_dims]
    X_test = np.load(test_path)    # [seq_len, n_dims]
    
    # 确保格式为 [seq_len, n_dims]
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    seq_len_test = X_test.shape[0]
    n_dims = X_test.shape[1]
    
    # 加载异常标注
    y_test = np.ones(seq_len_test, dtype=int)  # 默认全部正常
    
    try:
        with open(label_path, 'r') as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                if row['chan_id'].strip() == channel_name:
                    # 解析异常序列范围
                    anomaly_ranges = json.loads(row['anomaly_sequences'])
                    for start, end in anomaly_ranges:
                        if 0 <= start < seq_len_test and 0 <= end <= seq_len_test:
                            y_test[start:end] = 0  # 0=异常
                    break
    except Exception as e:
        print(f"警告：无法加载异常标注文件: {e}")
    
    # Z-score 标准化（各维度独立）
    X_train = z_normalize(X_train)
    X_test = z_normalize(X_test)
    
    print("\n" + "="*70)
    print(f"SMAP/MSL 数据集加载完成: {channel_name}")
    print("="*70)
    print(f"维度数: {n_dims}")
    print(f"训练集序列长度: {X_train.shape[0]}")
    print(f"测试集序列长度: {X_test.shape[0]}")
    print(f"测试集异常比例: {(1 - y_test.mean())*100:.1f}%")
    print(f"  - 正常时间步: {y_test.sum()}")
    print(f"  - 异常时间步: {len(y_test) - y_test.sum()}")
    
    return X_train, X_test, y_test


def get_smap_msl_channel_list(data_dir: str = "Datasets/SMAP&MSL") -> dict:
    """
    获取SMAP/MSL数据集中所有可用的通道列表及其统计信息
    
    返回：
        dict: {'SMAP': [...], 'MSL': [...]}
    """
    channels = {'SMAP': [], 'MSL': []}
    train_dir = os.path.join(data_dir, "train")
    
    if not os.path.exists(train_dir):
        return channels
    
    for file in os.listdir(train_dir):
        if file.endswith('.npy'):
            channel_name = file[:-4]
            # 根据前缀判断SMAP还是MSL
            spacecraft = "SMAP" if channel_name.startswith(('P-', 'S-', 'E-', 'A-', 'D-', 'F-', 'G-', 'M-', 'R-', 'T-', 'C-', 'B-')) else "MSL"
            
            # 更准确的判断：检查labeled_anomalies.csv
            try:
                label_path = os.path.join(data_dir, "labeled_anomalies.csv")
                with open(label_path, 'r') as f:
                    reader = csv_module.DictReader(f)
                    for row in reader:
                        if row['chan_id'].strip() == channel_name:
                            spacecraft = row['spacecraft'].strip()
                            break
            except:
                pass
            
            if spacecraft in channels:
                channels[spacecraft].append(channel_name)
    
    # 排序
    for key in channels:
        channels[key].sort()
    
    return channels
