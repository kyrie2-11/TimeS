# PSESC项目使用指南

## 🎯 项目概述

PSESC (Probabilistic Subsequence Early Classification) 项目基于 **Shapelet 隶属度的紧凑特征提取**方法，用于时间序列分类和异常检测任务。

## 📁 项目结构

```
PseSC_implementation/
├── models/                          # 核心模型代码
│   ├── decision_tree_classifier.py  # 决策树/随机森林分类器 ✨ 新增
│   ├── anomaly_detector.py          # 异常检测器 ✨ 新增
│   ├── data_loader.py               # 数据加载工具 ✨ 新增
│   ├── shapelet_pool.py             # Shapelet池构建
│   ├── membership_mapping.py        # 隶属度映射
│   ├── psesc_extractor.py           # PseSC特征提取
│   └── visualization.py             # 可视化工具
├── experiments/                     # 实验脚本
│   ├── ecg_classification_experiment.py    # ECG分类实验 ✨ 新增
│   └── ecg_anomaly_detection_experiment.py # ECG异常检测实验 ✨ 新增
├── Shapelet/                        # Shapelet发现模块 ✨ 新增
│   ├── __init__.py
│   └── mul_shapelet_discovery.py   # 简化的shapelet发现算法
├── Datasets/                        # 数据集
│   └── ECGFiveDays/                # ECG心电图数据集
├── results/                         # 实验结果
│   ├── classification/             # 分类结果
│   └── anomaly_detection/          # 异常检测结果
├── docs/                           # 文档
│   └── formula_reference.md        # 数学公式参考
├── run_experiments.py              # 主运行脚本 ✨ 新增
└── requirements.txt                # 依赖包列表
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行实验

#### 方法 A：使用交互式菜单

```bash
python run_experiments.py
```

选择要运行的实验：
- `[1]` ECG 心电图分类实验
- `[2]` ECG 心电图异常检测实验  
- `[3]` 运行所有实验

#### 方法 B：直接运行特定实验

**分类实验：**
```bash
python experiments\ecg_classification_experiment.py
```

**异常检测实验：**
```bash
python experiments\ecg_anomaly_detection_experiment.py
```

## 📊 实验说明

### 分类实验

在 ECGFiveDays 数据集上测试两种模型：

1. **决策树 (Decision Tree)**
   - PseSC特征提取
   - 单棵决策树分类
   - 可解释性强

2. **随机森林 (Random Forest)**
   - PseSC特征提取
   - 100棵树集成
   - 更高准确率

**输出结果：**
- `confusion_matrix_*.png` - 混淆矩阵
- `feature_importance_*.png` - 特征重要性图
- `decision_tree_viz.png` - 决策树可视化
- `model_comparison.png` - 模型性能对比
- `classification_results.csv` - 结果汇总

### 异常检测实验

使用正常样本训练，检测异常心电图信号：

1. **Elliptic Envelope (Gaussian)**
   - 基于高斯分布
   - 马氏距离计算异常分数

2. **One-Class SVM**
   - 基于支持向量机
   - RBF核函数

**输出结果：**
- `anomaly_curves_*.png` - ROC/PR曲线
- `anomaly_heatmap_*.png` - 异常热力图（可解释性）
- `normal_heatmap_*.png` - 正常样本热力图（对比）
- `method_comparison.png` - 方法性能对比
- `anomaly_detection_results.csv` - 结果汇总

## 🔬 核心方法原理

### PSESC 特征提取流程

1. **Shapelet 池构建**
   - 从每个类别提取最具区分度的子序列
   - 使用信息增益评估候选片段
   - 默认每类3个shapelet

2. **隶属度映射**
   ```
   μ_t = exp(-β_s * d_t)
   ```
   - `d_t`: 时间位置t的距离
   - `β_s`: 自适应尺度参数
   - `μ_t`: 隶属度值 ∈ (0, 1]

3. **PseSC特征提取**（每个shapelet生成6维特征）
   - **时间统计量**：
     - `c1`: 质心（加权平均位置）
     - `c2`: 方差（时间分散程度）
     - `c3`: 偏度（分布对称性）
   - **顺序统计量**：
     - `θ1, θ2, θ3`: Top-3 隶属度值

4. **最终特征向量**
   ```
   PseSC = [c1, c2, c3, θ1, θ2, θ3] × M个shapelet = 6M维
   ```

### 优势

✅ **信息丰富**：保留时空分布信息（而非仅最小距离）  
✅ **定长向量**：无论序列多长，特征维度固定  
✅ **可解释性强**：每个特征有明确物理意义  
✅ **分类器友好**：可用任何表格数据分类算法  

## 📈 实验结果示例

运行完成后，在 `results/` 目录下查看：

```
results/
├── classification/
│   ├── confusion_matrix_decisiontree.png
│   ├── confusion_matrix_randomforest.png
│   ├── feature_importance_decisiontree.png
│   ├── feature_importance_randomforest.png
│   ├── decision_tree_viz.png
│   ├── model_comparison.png
│   └── classification_results.csv
└── anomaly_detection/
    ├── anomaly_curves_gaussian.png
    ├── anomaly_curves_svm.png
    ├── anomaly_heatmap_sample_*.png
    ├── normal_heatmap_sample_*.png
    ├── method_comparison.png
    └── anomaly_detection_results.csv
```

## 🛠️ 自定义实验

### 调整 Shapelet 数量

```python
classifier = PseSCDecisionTreeClassifier(
    n_shapelets_per_class=5,  # 每类5个shapelet（默认3）
    beta_mode='adaptive',
    use_random_forest=True
)
```

### 调整决策树参数

```python
tree_params = {
    'max_depth': 12,           # 最大深度
    'min_samples_split': 10,   # 最小分裂样本数
    'n_estimators': 200        # 随机森林树的数量
}

classifier = PseSCDecisionTreeClassifier(
    tree_params=tree_params,
    use_random_forest=True
)
```

### 调整异常检测灵敏度

```python
detector = PseSCAnomalyDetector(
    contamination=0.05,  # 预期异常比例（默认0.1）
    method='gaussian'    # 或 'svm'
)
```

## 📝 论文写作建议

### 创新点描述

1. **核心创新**：基于隶属度分布的 Shapelet 特征增强范式
   - 将距离序列映射为隶属度分布
   - 提取时空统计量作为紧凑表示

2. **方法优势**：
   - 信息保真度高（vs 传统Shapelet只保留最小距离）
   - 可解释性强（vs 深度学习黑盒模型）
   - 定长表示（vs Shapelet Transform高维可变长度）

3. **适用场景**：
   - 医疗健康（心电图、脑电图异常检测）
   - 金融风控（交易序列异常识别）
   - 工业监控（设备状态分类与故障预测）

### 实验设置建议

- **基线对比**：
  - 传统 Shapelet Transform
  - ROCKET / MiniROCKET
  - 1D-CNN / LSTM

- **消融实验**：
  - 仅使用时间统计量 [c1, c2, c3]
  - 仅使用顺序统计量 [θ1, θ2, θ3]
  - 完整 PseSC 向量

- **参数敏感性**：
  - Shapelet 数量 M 的影响
  - β 模式（自适应 vs 全局）
  - 窗口大小的影响

## 🐛 常见问题

### Q1: 运行时提示 "ModuleNotFoundError"
**A**: 确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

### Q2: 实验运行很慢
**A**: 这是正常的。Shapelet发现需要计算大量距离矩阵。可以：
- 减少 `n_shapelets_per_class`
- 减少训练样本数
- 使用更快的shapelet算法（如tslearn）

### Q3: 如何在自己的数据集上运行？
**A**: 
1. 将数据集放入 `Datasets/YourDataset/`
2. 格式：第一列为标签，其余列为时间序列值（逗号分隔）
3. 修改实验脚本中的 `dataset_name='YourDataset'`

## 📚 参考文献

- Shapelet Transform原论文
- 时间序列分类综述
- UCR Time Series Archive

## 👨‍💻 开发者

基于 Shapelet 隶属度的紧凑特征提取方法

---

**祝实验顺利！如有问题欢迎提Issue。** 🎉
