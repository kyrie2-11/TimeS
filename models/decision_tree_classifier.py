"""
基于 PseSC 特征的决策树分类器
用于时间序列分类和异常检测任务
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

from models.membership_mapping import compute_memberships
from models.psesc_extractor import extract_psesc_features
from models.shapelet_pool import build_shapelet_pool


class PseSCDecisionTreeClassifier:
    """
    基于 PseSC 特征的决策树分类器封装类
    
    工作流程：
    1. 从训练集构建 Shapelet 池
    2. 提取 PseSC 特征 (6M 维向量)
    3. 训练决策树/随机森林分类器
    4. 评估与可视化
    """
    
    def __init__(self, n_shapelets_per_class=3, beta_mode='adaptive', 
                 tree_params=None, use_random_forest=False):
        """
        参数：
            n_shapelets_per_class: 每个类别提取的 shapelet 数量
            beta_mode: 隶属度映射的 β 模式 ('adaptive' 或 'global')
            tree_params: 决策树超参数字典
            use_random_forest: 是否使用随机森林（更强大，防止过拟合）
        """
        self.n_shapelets_per_class = n_shapelets_per_class
        self.beta_mode = beta_mode
        self.shapelets = None
        self.shapelets_info = None
        self.scaler = StandardScaler()
        
        # 决策树默认参数（防止过拟合）
        default_params = {
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        if tree_params:
            default_params.update(tree_params)
        
        # 选择模型
        if use_random_forest:
            default_params['n_estimators'] = default_params.get('n_estimators', 100)
            self.model = RandomForestClassifier(**default_params)
            self.model_type = 'RandomForest'
        else:
            self.model = DecisionTreeClassifier(**default_params)
            self.model_type = 'DecisionTree'
    
    def _format_dataset(self, time_series_list):
        """
        将时间序列列表转换为 (n_samples, n_dims, seq_len) 格式
        
        参数：
            time_series_list: 时间序列列表
        
        返回：
            formatted_data: (n_samples, n_dims, seq_len)
        """
        n_samples = len(time_series_list)
        
        # 检查第一个样本的形状
        first_ts = time_series_list[0]
        if first_ts.ndim == 1:
            # 单变量时间序列
            seq_len = len(first_ts)
            n_dims = 1
            formatted_data = np.zeros((n_samples, n_dims, seq_len))
            for i, ts in enumerate(time_series_list):
                formatted_data[i, 0, :] = ts
        else:
            # 多变量时间序列
            if first_ts.shape[0] < first_ts.shape[1]:
                # 已经是 (n_dims, seq_len) 格式
                n_dims, seq_len = first_ts.shape
            else:
                # (seq_len, n_dims) 格式，需要转置
                seq_len, n_dims = first_ts.shape
            
            formatted_data = np.zeros((n_samples, n_dims, seq_len))
            for i, ts in enumerate(time_series_list):
                if ts.shape[0] < ts.shape[1]:
                    formatted_data[i, :, :] = ts
                else:
                    formatted_data[i, :, :] = ts.T
        
        return formatted_data
    
    def _extract_features_batch(self, time_series_list):
        """
        批量提取 PseSC 特征
        
        参数：
            time_series_list: 时间序列列表，每个元素 shape [L] (单变量)
        
        返回：
            features: [N, 6M] 特征矩阵
        """
        features = []
        for ts in time_series_list:
            # 转换为 (n_dims, seq_len) 格式，单变量时 n_dims=1
            if ts.ndim == 1:
                ts_formatted = ts.reshape(1, -1)  # (1, L)
            else:
                ts_formatted = ts.T if ts.shape[0] > ts.shape[1] else ts
            
            # 计算隶属度
            memberships_dict = compute_memberships(
                ts_formatted, 
                self.shapelets, 
                self.shapelets_info,
                beta_mode=self.beta_mode
            )
            
            # 提取 PseSC 特征
            psesc_vec = extract_psesc_features(memberships_dict)
            features.append(psesc_vec)
        
        return np.array(features)
    
    def fit(self, X_train, y_train):
        """
        训练分类器
        
        参数：
            X_train: 训练时间序列列表 [N个样本]
            y_train: 训练标签 [N]
        
        返回：
            self
        """
        print("\n" + "="*60)
        print(f"训练 {self.model_type} 分类器")
        print("="*60)
        
        print(f"步骤 1/3: 构建 Shapelet 池...")
        
        # 转换数据格式为 (n_samples, n_dims, seq_len)
        X_train_formatted = self._format_dataset(X_train)
        y_train_array = np.array(y_train)
        
        self.shapelets, self.shapelets_info = build_shapelet_pool(
            X_train_formatted, y_train_array, 
            n_per_class=self.n_shapelets_per_class
        )
        
        M = sum(len(s_list) for s_list in self.shapelets)
        print(f"  → 提取了 {M} 个 shapelets")
        
        print(f"步骤 2/3: 提取 PseSC 特征...")
        X_features = self._extract_features_batch(X_train)
        print(f"  → 特征维度: {X_features.shape} (每个样本 {X_features.shape[1]} 维)")
        
        # 特征标准化
        X_features = self.scaler.fit_transform(X_features)
        
        print(f"步骤 3/3: 训练{self.model_type}模型...")
        self.model.fit(X_features, y_train)
        
        # 训练集准确率
        train_acc = self.model.score(X_features, y_train)
        print(f"  ✓ 训练完成！训练集准确率: {train_acc:.4f}")
        
        return self
    
    def predict(self, X_test):
        """预测类别"""
        X_features = self._extract_features_batch(X_test)
        X_features = self.scaler.transform(X_features)
        return self.model.predict(X_features)
    
    def predict_proba(self, X_test):
        """预测概率（用于异常检测或置信度分析）"""
        X_features = self._extract_features_batch(X_test)
        X_features = self.scaler.transform(X_features)
        return self.model.predict_proba(X_features)
    
    def evaluate(self, X_test, y_test, plot_cm=True, save_path='./'):
        """
        全面评估模型性能
        
        参数：
            X_test: 测试时间序列
            y_test: 测试标签
            plot_cm: 是否绘制混淆矩阵
            save_path: 图片保存路径
        
        返回：
            metrics: 包含各种评估指标的字典
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # 打印分类报告
        print("\n" + "="*60)
        print("分类报告：")
        print("="*60)
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        if plot_cm:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       cbar_kws={'label': 'Count'})
            plt.title(f'Confusion Matrix - {self.model_type}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_path = f'{save_path}/confusion_matrix_{self.model_type.lower()}.png'
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ 混淆矩阵已保存为 {cm_path}")
            plt.close()
        
        return metrics
    
    def get_feature_importance(self, top_k=20, save_path='./'):
        """
        获取特征重要性（用于可解释性分析）
        
        参数：
            top_k: 显示前 k 个最重要特征
            save_path: 图片保存路径
        
        返回：
            importance_df: 按重要性排序的 DataFrame
        """
        if isinstance(self.model, RandomForestClassifier):
            importances = self.model.feature_importances_
        else:
            importances = self.model.feature_importances_
        
        # 解析特征名称（shapelet索引 + 统计量类型）
        M = sum(len(s_list) for s_list in self.shapelets)
        feature_names = []
        stat_names = ['c1_centroid', 'c2_variance', 'c3_skewness', 
                     'θ1_top1', 'θ2_top2', 'θ3_top3']
        for i in range(M):
            for stat in stat_names:
                feature_names.append(f"S{i}_{stat}")
        
        # 创建 DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_k)
        
        print(f"\nTop {top_k} 最重要特征:")
        print(importance_df.to_string(index=False))
        
        # 可视化
        plt.figure(figsize=(10, max(6, top_k * 0.3)))
        plt.barh(range(top_k), importance_df['Importance'].values[::-1], 
                color='steelblue')
        plt.yticks(range(top_k), importance_df['Feature'].values[::-1])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Most Important Features - {self.model_type}')
        plt.tight_layout()
        importance_path = f'{save_path}/feature_importance_{self.model_type.lower()}.png'
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        print(f"✓ 特征重要性图已保存为 {importance_path}")
        plt.close()
        
        return importance_df
    
    def visualize_tree(self, max_depth=3, save_path='./'):
        """
        可视化决策树（仅适用于单棵树）
        
        参数：
            max_depth: 显示的最大深度
            save_path: 图片保存路径
        """
        if isinstance(self.model, RandomForestClassifier):
            print("随机森林包含多棵树，仅可视化第一棵树...")
            tree_to_plot = self.model.estimators_[0]
        else:
            tree_to_plot = self.model
        
        plt.figure(figsize=(20, 10))
        plot_tree(tree_to_plot, max_depth=max_depth, 
                 filled=True, fontsize=10, rounded=True)
        plt.title(f'Decision Tree Visualization (max_depth={max_depth})')
        plt.tight_layout()
        tree_path = f'{save_path}/decision_tree_viz.png'
        plt.savefig(tree_path, dpi=150, bbox_inches='tight')
        print(f"✓ 决策树可视化已保存为 {tree_path}")
        plt.close()
