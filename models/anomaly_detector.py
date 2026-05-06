"""
基于 PseSC 特征的异常检测器
用于时间序列全局异常检测和异常定位
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report

from models.membership_mapping import MembershipMapper
from models.psesc_extractor import PseSCExtractor
from models.shapelet_pool import build_shapelet_pool
from models.anomaly_metrics import compute_pointwise_metrics, compute_event_metrics


class PseSCAnomalyDetector:
    """基于多尺度 PseSC 特征的异常检测器。"""

    def __init__(
        self,
        n_shapelets_per_class=3,
        beta_mode='adaptive',
        contamination=0.1,
        method='svm',
        random_state=42,
        window_sizes=(20, 30, 40),
        use_multichannel=True,
        channel_weight_mode='inverse_mad',
        one_class_shapelet_mode='auto',
    ):
        self.n_shapelets_per_class = n_shapelets_per_class
        self.beta_mode = beta_mode
        self.contamination = contamination
        self.method = method
        self.random_state = random_state

        self.window_sizes = tuple(window_sizes) if window_sizes is not None else (30,)
        self.use_multichannel = use_multichannel
        self.channel_weight_mode = channel_weight_mode
        self.one_class_shapelet_mode = one_class_shapelet_mode

        self.scaler = StandardScaler()
        self.psesc_extractor = PseSCExtractor(n_order_stats=3)
        self.score_threshold = None

        self.shapelet_pools = []
        self.mapper = None
        self.shapelets = None
        self.shapelets_info = None

        self.detector = self._init_detector()

    def _init_detector(self):
        if self.method == 'iforest':
            return IsolationForest(
                n_estimators=300,
                contamination=self.contamination,
                random_state=self.random_state,
                max_samples='auto',
            )
        if self.method == 'gaussian':
            return EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state,
            )

        return OneClassSVM(
            nu=self.contamination,
            kernel='rbf',
            gamma='scale',
        )

    def fit(self, X_normal):
        print('\n' + '=' * 60)
        print(f'训练异常检测器 (方法: {self.method.upper()})')
        print('=' * 60)
        print(f'使用 {len(X_normal)} 个正常样本训练...')

        dummy_labels = np.zeros(len(X_normal), dtype=int)
        X_normal_formatted = self._format_dataset(X_normal)

        self.shapelet_pools = []
        print('步骤 1/2: 构建多尺度 Shapelet 池...')
        for w in self.window_sizes:
            shapelets, shapelets_info = build_shapelet_pool(
                X_normal_formatted,
                dummy_labels,
                n_per_class=self.n_shapelets_per_class,
                window_size=w,
                one_class_mode=self.one_class_shapelet_mode,
                random_state=self.random_state,
            )

            mapper = MembershipMapper(
                shapelets,
                shapelets_info,
                beta_mode=self.beta_mode,
                use_multichannel=self.use_multichannel,
            )

            mapper.learn_channel_weights(X_normal_formatted, mode=self.channel_weight_mode)
            if self.beta_mode == 'adaptive':
                mapper.calibrate_adaptive_beta(X_normal_formatted)

            self.shapelet_pools.append(
                {
                    'window_size': w,
                    'shapelets': shapelets,
                    'shapelets_info': shapelets_info,
                    'mapper': mapper,
                }
            )

        # 兼容旧代码接口
        if len(self.shapelet_pools) > 0:
            self.mapper = self.shapelet_pools[0]['mapper']
            self.shapelets = self.shapelet_pools[0]['shapelets']
            self.shapelets_info = self.shapelet_pools[0]['shapelets_info']

        print(f"  → 共构建 {len(self.shapelet_pools)} 个尺度: {self.window_sizes}")

        print('步骤 2/2: 提取 PseSC 特征并训练检测器...')
        X_features = self._extract_features_batch(X_normal)
        X_features = self.scaler.fit_transform(X_features)
        print(f'  → 特征维度: {X_features.shape}')

        self.detector.fit(X_features)

        train_scores = self._score_from_scaled_features(X_features)
        self.score_threshold = np.percentile(train_scores, self.contamination * 100.0)
        print('  ✓ 异常检测器训练完成！')

        return self

    def _format_dataset(self, time_series_list):
        n_samples = len(time_series_list)
        first_ts = time_series_list[0]

        if first_ts.ndim == 1:
            seq_len = len(first_ts)
            n_dims = 1
            formatted_data = np.zeros((n_samples, n_dims, seq_len))
            for i, ts in enumerate(time_series_list):
                formatted_data[i, 0, :] = ts
        else:
            if first_ts.shape[0] < first_ts.shape[1]:
                n_dims, seq_len = first_ts.shape
            else:
                seq_len, n_dims = first_ts.shape

            formatted_data = np.zeros((n_samples, n_dims, seq_len))
            for i, ts in enumerate(time_series_list):
                if ts.shape[0] < ts.shape[1]:
                    formatted_data[i, :, :] = ts
                else:
                    formatted_data[i, :, :] = ts.T

        return formatted_data

    def _extract_features_batch(self, time_series_list):
        features = []
        for ts in time_series_list:
            features.append(self._extract_features_single(ts))
        return np.array(features)

    def _to_dim_first(self, ts):
        if ts.ndim == 1:
            return ts.reshape(1, -1)
        return ts.T if ts.shape[0] > ts.shape[1] else ts

    def _extract_features_single(self, ts):
        ts_dim_first = self._to_dim_first(ts)
        ts_1d = ts if ts.ndim == 1 else np.mean(ts_dim_first, axis=0)

        if len(self.shapelet_pools) == 0:
            raise RuntimeError('检测器尚未训练，shapelet 池不可用')

        all_scale_features = []
        for pool in self.shapelet_pools:
            mapper = pool['mapper']
            memberships_dict = mapper.compute_memberships(ts_dim_first)

            psesc_vec = self.psesc_extractor.extract_full_psesc(memberships_dict)
            membership_stats = self._extract_membership_stats(memberships_dict)
            all_scale_features.append(np.concatenate([psesc_vec, membership_stats]))

        amplitude_stats = np.array([
            np.mean(ts_1d),
            np.std(ts_1d),
            np.max(ts_1d) - np.min(ts_1d),
            np.sqrt(np.mean(np.square(ts_1d))),
            np.max(np.abs(ts_1d)),
        ])

        return np.concatenate(all_scale_features + [amplitude_stats])

    def _extract_membership_stats(self, memberships_dict):
        stats = []
        for class_key in sorted(memberships_dict.keys()):
            class_dict = memberships_dict[class_key]
            for shapelet_key in sorted(class_dict.keys()):
                data = class_dict[shapelet_key]
                mu = data['memberships']
                dist = data['distances']
                mu_sorted = np.sort(mu)[::-1]
                top1 = mu_sorted[0] if len(mu_sorted) > 0 else 0.0
                top2 = mu_sorted[1] if len(mu_sorted) > 1 else 0.0
                stats.extend([
                    np.max(mu),
                    np.mean(mu),
                    np.std(mu),
                    top1 - top2,
                    np.min(dist),
                ])
        return np.array(stats, dtype=float)

    def _score_from_scaled_features(self, X_features_scaled):
        if hasattr(self.detector, 'decision_function'):
            return self.detector.decision_function(X_features_scaled)
        if hasattr(self.detector, 'score_samples'):
            return self.detector.score_samples(X_features_scaled)
        pred = self.detector.predict(X_features_scaled)
        return pred.astype(float)

    def predict(self, X_test):
        X_features = self._extract_features_batch(X_test)
        X_features = self.scaler.transform(X_features)
        scores = self._score_from_scaled_features(X_features)

        if self.score_threshold is None:
            return self.detector.predict(X_features)

        return np.where(scores >= self.score_threshold, 1, -1)

    def anomaly_score(self, X_test):
        X_features = self._extract_features_batch(X_test)
        X_features = self.scaler.transform(X_features)
        return self._score_from_scaled_features(X_features)

    def evaluate(self, X_test, y_test, save_path='./'):
        predictions = self.predict(X_test)
        scores = self.anomaly_score(X_test)

        predictions_binary = (predictions == 1).astype(int)  # 1 normal, 0 anomaly
        accuracy = float((predictions_binary == y_test).mean())

        # anomaly=1 形式用于工业指标
        y_true_anomaly = (1 - y_test).astype(int)
        y_pred_anomaly = (1 - predictions_binary).astype(int)

        point_metrics = compute_pointwise_metrics(y_true_anomaly, y_pred_anomaly)
        event_metrics = compute_event_metrics(y_true_anomaly, y_pred_anomaly)

        print('\n' + '=' * 60)
        print('异常检测评估报告')
        print('=' * 60)
        print(f'准确率: {accuracy:.4f}')
        print('\n详细分类报告:')
        print(
            classification_report(
                y_test,
                predictions_binary,
                target_names=['异常', '正常'],
                zero_division=0,
            )
        )

        metrics = self._plot_roc_pr_curves(y_true_anomaly, scores, save_path)
        metrics['accuracy'] = accuracy
        metrics.update(point_metrics)
        metrics.update(event_metrics)

        return metrics

    def _plot_roc_pr_curves(self, y_true_anomaly, scores, save_path):
        # 数值越小越异常，所以取负号后分数越大越异常
        anomaly_scores = -scores

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        fpr, tpr, _ = roc_curve(y_true_anomaly, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2, color='darkorange')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title(f'ROC Curve ({self.method.upper()})', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)

        precision, recall, _ = precision_recall_curve(y_true_anomaly, anomaly_scores)
        ax2.plot(recall, precision, linewidth=2, color='green')
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title(f'Precision-Recall Curve ({self.method.upper()})', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        curve_path = f'{save_path}/anomaly_curves_{self.method}.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        print(f'\n✓ ROC/PR 曲线已保存为 {curve_path}')
        plt.close()

        return {'auc': float(roc_auc)}

    def _resample_curve(self, curve, target_len):
        if len(curve) == target_len:
            return curve
        x_old = np.linspace(0.0, 1.0, len(curve))
        x_new = np.linspace(0.0, 1.0, target_len)
        return np.interp(x_new, x_old, curve)

    def locate_anomaly(self, time_series, threshold_percentile=95):
        ts_dim_first = self._to_dim_first(time_series)
        seq_len = ts_dim_first.shape[1]

        if len(self.shapelet_pools) == 0:
            raise RuntimeError('检测器尚未训练，无法定位异常')

        anomaly_curves = []
        for pool in self.shapelet_pools:
            mapper = pool['mapper']
            memberships_dict = mapper.compute_memberships(ts_dim_first)

            all_memberships = []
            weights = []
            for class_key in sorted(memberships_dict.keys()):
                class_dict = memberships_dict[class_key]
                for shapelet_key in sorted(class_dict.keys()):
                    sd = class_dict[shapelet_key]
                    all_memberships.append(sd['memberships'])
                    weights.append(max(0.0, float(sd.get('info_gain', 0.0))))

            memberships = np.array(all_memberships)
            w = np.array(weights, dtype=float)
            if np.sum(w) < 1e-8:
                w = np.ones_like(w)
            w = w / np.sum(w)

            normality_curve = np.dot(w, memberships)
            anomaly_curve = 1.0 - normality_curve
            anomaly_curves.append(self._resample_curve(anomaly_curve, seq_len))

        fused_anomaly_curve = np.mean(np.array(anomaly_curves), axis=0)
        threshold = np.percentile(fused_anomaly_curve, threshold_percentile)
        suspicious_positions = np.where(fused_anomaly_curve >= threshold)[0]

        anomaly_positions = [
            {
                'time_position': int(pos),
                'anomaly_intensity': float(fused_anomaly_curve[pos]),
            }
            for pos in suspicious_positions
        ]

        return {
            'anomaly_positions': anomaly_positions,
            'anomaly_curve': fused_anomaly_curve,
        }

    def visualize_anomaly_heatmap(self, time_series, save_path='./anomaly_heatmap.png'):
        ts_dim_first = self._to_dim_first(time_series)
        time_series_plot = time_series if time_series.ndim == 1 else np.mean(ts_dim_first, axis=0)

        if len(self.shapelet_pools) == 0:
            raise RuntimeError('检测器尚未训练，无法可视化异常')

        # 用首个尺度展示热力图，避免不同窗口导致列数不一致
        memberships_dict = self.shapelet_pools[0]['mapper'].compute_memberships(ts_dim_first)

        all_memberships = []
        for class_key in sorted(memberships_dict.keys()):
            class_dict = memberships_dict[class_key]
            for shapelet_key in sorted(class_dict.keys()):
                all_memberships.append(class_dict[shapelet_key]['memberships'])

        memberships = np.array(all_memberships)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        ax1.plot(time_series_plot, linewidth=1.5, color='black')
        ax1.set_title('Original Time Series', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.grid(alpha=0.3)

        im = ax2.imshow(memberships, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax2.set_title('Membership Heatmap (Scale-1)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Position')
        ax2.set_ylabel('Shapelet Index')
        plt.colorbar(im, ax=ax2, label='Membership μ')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✓ 异常热力图已保存为 {save_path}')
        plt.close()
