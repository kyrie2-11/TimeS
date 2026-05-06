# PseSC 实现文档

## 概述

本文件夹包含 PseSC (Pseudo Shapelet-based Compact) 特征提取的完整实现，用于可解释的时间序列分类。

## 核心组件

### 1. Shapelet 池构建 (`shapelet_pool.py`)
- OSD 离线提取按类别构建的 shapelet 池
- z-normalize 预处理
- BasicMotions 默认配置：M=12（每类 3 个）

### 2. 滑窗匹配与隶属度映射 (`membership_mapping.py`)
- 多通道距离计算
- 自适应尺度 β_s 或全局常数 β=2.0
- 指数映射：μ_t = exp(-β_s * d_t)

### 3. PseSC 特征提取 (`psesc_extractor.py`)
- **时间统计量** c = [c1, c2, c3]（质心、方差、偏度）
- **顺序统计量** θ = [θ1, θ2, θ3]（top-3 强度）
- 数值稳健处理

### 4. 可视化工具 (`visualization.py`)
- 隶属度热力图
- 序列叠加图
- 分类得分可视化

### 5. 完整示例 (`demo.py`)
- 端到端 pipeline 演示
- BasicMotions 数据集示例

## 数学定义

### Shapelet 池

给定多元时间序列 x ∈ R^(L×D) 和 shapelet 集合 S = {s_j}_{j=1}^M，
每个 s_j 长度为 l_j，使用 OSD 离线提取。

### 距离计算

对长度 l 的 shapelet s，滑窗数 T = L - l + 1，每位置 t ∈ {1,...,T}：

```
d_t = sqrt(Σ_{d=1}^D w_d ||x_{t:t+l-1}^(d) - s^(d)||_2^2)
```

其中 Σ_d w_d = 1，默认 w_d = 1/D

### 隶属度映射

```
μ_t = exp(-β_s * d_t), μ_t ∈ (0, 1]
```

自适应尺度：
```
m_s = median{min_t d_t(x_i, s)}_{x_i ∈ train}
β_s = ln(2) / m_s
```

或使用全局常数 β = 2.0

### PseSC-Time 特征

位置归一化：p_t = (t - 0.5) / T
总隶属度：S = Σ_{t=1}^T μ_t
数值稳健项：ε = 1e-8

**时间统计量：**
```
c1 = Σ_t (p_t * μ_t) / S          # 质心
c2 = Σ_t (p_t - c1)^2 * μ_t / S  # 方差
c3 = Σ_t (p_t - c1)^3 * μ_t / ((c2 + ε)^3 * S)  # 偏度
```

**顺序统计量：**
```
θ1 ≥ θ2 ≥ θ3  # {μ_t} 的前三大值
```

**单个 shapelet 的表征：**
```
PseSC_time(s) = [c1, c2, c3, θ1, θ2, θ3]^T ∈ R^6
```

**数值兜底：**
若 S < 1e-6，取 c1 = 0.5, c2 = c3 = 0, θ_{1..3} = 0

## 文件结构

```
PseSC_implementation/
├── README.md                   # 本文档
├── shapelet_pool.py           # Shapelet 池构建
├── membership_mapping.py      # 隶属度映射实现
├── psesc_extractor.py         # PseSC 特征提取器
├── visualization.py           # 可视化工具
├── demo.py                    # 完整示例
├── requirements.txt           # 依赖列表
└── docs/
    ├── formula_reference.md   # 公式详细说明
    ├── api_reference.md       # API 文档
    └── examples.md            # 使用示例
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
python demo.py
```

### 使用示例

```python
from shapelet_pool import build_shapelet_pool
from membership_mapping import compute_memberships
from psesc_extractor import extract_psesc_features

# 1. 构建 shapelet 池
shapelets = build_shapelet_pool(train_data, train_labels, n_per_class=3)

# 2. 计算隶属度
memberships = compute_memberships(test_sample, shapelets, beta_mode='adaptive')

# 3. 提取 PseSC 特征
features = extract_psesc_features(memberships)

# 4. 分类
predictions = classifier.predict(features)
```

## 参数配置

### BasicMotions 默认配置
- 数据集：32 训练样本 / 40 测试样本
- 类别数：4
- 维度：6
- 序列长度：100
- Shapelet 数：M=12（每类 3 个）
- β 模式：adaptive（自适应）或 β=2.0（全局常数）

## 参考文献

1. Ye, L., & Keogh, E. (2009). Time series shapelets: a new primitive for data mining.
2. Hills, J., et al. (2014). Classification of time series by shapelet transformation.
3. Zadeh, L. A. (1965). Fuzzy sets.

## 许可证

本项目遵循 MIT 许可证。
