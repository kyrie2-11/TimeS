"""
工业数据集异常检测实验模板
========================
支持 SWaT / SMD / MSL 的统一实验入口。

用法示例：
python experiments/industrial_anomaly_experiment.py --dataset swat --data-dir D:/data/SWaT
python experiments/industrial_anomaly_experiment.py --dataset smd --data-dir D:/data/SMD
python experiments/industrial_anomaly_experiment.py --dataset msl --data-dir D:/data/MSL
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import PseSCAnomalyDetector


def to_sequence_list(x_2d_or_3d, window=120, stride=10):
    """
    将连续数据切分为序列样本。
    输入:
        - 2D: (T, D)
        - 3D: (N, D, L) 直接转 list
    """
    if x_2d_or_3d.ndim == 3:
        return [x_2d_or_3d[i] for i in range(x_2d_or_3d.shape[0])]

    x = x_2d_or_3d
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    seqs = []
    for i in range(0, len(x) - window + 1, stride):
        seg = x[i:i + window].T  # (D, L)
        seqs.append(seg)
    return seqs


def label_to_window_label(label, window=120, stride=10):
    """点级标签转窗口标签：窗口内任一点异常则窗口异常。"""
    ys = []
    for i in range(0, len(label) - window + 1, stride):
        ys.append(0 if np.any(label[i:i + window] == 1) else 1)  # 1 normal, 0 anomaly
    return np.array(ys, dtype=int)


def load_swat(data_dir):
    train_path = os.path.join(data_dir, 'swat_train.csv')
    test_path = os.path.join(data_dir, 'swat_test.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_col = 'label' if 'label' in test_df.columns else test_df.columns[-1]
    y_test_point = test_df[label_col].to_numpy().astype(int)

    x_train = train_df.drop(columns=[c for c in ['label', 'Label'] if c in train_df.columns]).to_numpy(dtype=float)
    x_test = test_df.drop(columns=[label_col]).to_numpy(dtype=float)

    return x_train, x_test, y_test_point


def load_npy_triplet(data_dir):
    x_train = np.load(os.path.join(data_dir, 'train.npy'))
    x_test = np.load(os.path.join(data_dir, 'test.npy'))
    y_test_point = np.load(os.path.join(data_dir, 'test_label.npy')).astype(int)
    return x_train, x_test, y_test_point


def run_single_experiment(dataset, data_dir, args):
    if dataset == 'swat':
        x_train, x_test, y_test_point = load_swat(data_dir)
    elif dataset in ('smd', 'msl'):
        x_train, x_test, y_test_point = load_npy_triplet(data_dir)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    X_normal_list = to_sequence_list(x_train, window=args.segment_length, stride=args.segment_stride)
    X_test_list = to_sequence_list(x_test, window=args.segment_length, stride=args.segment_stride)
    y_test = label_to_window_label(y_test_point, window=args.segment_length, stride=args.segment_stride)

    detector = PseSCAnomalyDetector(
        n_shapelets_per_class=args.n_shapelets,
        beta_mode='adaptive',
        contamination=args.contamination,
        method=args.method,
        window_sizes=tuple(args.window_sizes),
        use_multichannel=True,
        channel_weight_mode='inverse_mad',
        one_class_shapelet_mode='auto',
    )

    detector.fit(X_normal_list)
    metrics = detector.evaluate(X_test_list, y_test, save_path=args.output_dir)

    print('\n===== Metrics Summary =====')
    for k in ['accuracy', 'auc', 'point_precision', 'point_recall', 'point_f1', 'event_precision', 'event_recall', 'event_f1']:
        if k in metrics:
            print(f'{k}: {metrics[k]:.4f}')


def run_smd_per_machine(data_dir, args):
    """
    SMD 推荐按 machine 单独建模。
    目录结构示例:
    data_dir/
      machine-1-1/
        train.npy test.npy test_label.npy
      machine-1-2/
        ...
    """
    machine_dirs = [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    all_metrics = []
    for md in machine_dirs:
        print('\n' + '=' * 60)
        print(f'Running machine: {os.path.basename(md)}')
        print('=' * 60)
        run_single_experiment('smd', md, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['swat', 'smd', 'msl'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./results/anomaly_detection')

    parser.add_argument('--method', type=str, default='svm', choices=['svm', 'iforest', 'gaussian'])
    parser.add_argument('--contamination', type=float, default=0.1)
    parser.add_argument('--n-shapelets', type=int, default=6)

    parser.add_argument('--segment-length', type=int, default=120)
    parser.add_argument('--segment-stride', type=int, default=10)
    parser.add_argument('--window-sizes', type=int, nargs='+', default=[20, 30, 40])

    parser.add_argument('--smd-per-machine', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == 'smd' and args.smd_per_machine:
        run_smd_per_machine(args.data_dir, args)
    else:
        run_single_experiment(args.dataset, args.data_dir, args)


if __name__ == '__main__':
    main()
