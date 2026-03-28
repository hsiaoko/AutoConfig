# AutoConfig - 图查询执行时间预测

自动配置系统 - 基于机器学习的图查询执行时间预测

## 项目概述

本项目旨在为任意图查询给定查询 Q、数据图 G 以及系统配置 Conf，预测该图查询的执行时间。

## 系统架构

本任务分为三个子系统：

### 1. 特征提取 (Feature Extraction)

采用**三阶段特征提取方法**，支持任意 Hybrid 图查询程序：

| 阶段 | 特征类型 | 特征数 | 来源 |
|------|---------|--------|------|
| **静态分析** | 代码结构特征 | 8 | 查询源代码 |
| **符号匹配** | 工作负载模板 | 12 | 代码模式 + 图统计 |
| **图感知实例化** | 图/分区统计 | 17 | 图 G + 分区 F |
| **配置参数** | 系统配置 | 10 | 配置字典 |

**静态特征**（8 个）：
- 循环计数、最大循环深度
- 分支计数、变量计数
- 递归计数、原子操作计数
- 同步计数、显式并行标志

**符号特征**（6 个模板，12 个特征）：
- **VScan**: 顶点扫描 (|V|)
- **EScan**: 边扫描 (|E|)
- **FScan**: 前沿迭代 (∑|E_t|)
- **RExp**: 递归扩展 (∏deg)
- **Atom**: 原子更新 (|E|×skew)
- **Comm**: 跨分区通信 (∑deg_∂)

### 2. 模型训练 (Model Training)

训练贝叶斯浅层模型用于预测执行时间：

- **贝叶斯岭回归**: 自动正则化参数调优
- **不确定性估计**: 提供预测置信区间
- **对数变换**: 处理偏态时间分布
- **抗过拟合**: 适合小样本训练

### 3. 预测阶段 (Prediction)

给定 Q, G, Config 预测执行代价：

- 单次/批量预测
- 不确定性量化
- 特征重要性分析
- 模型保存/加载

## 安装

```bash
cd autoconfig
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 快速开始

### 运行示例

```bash
# 训练和预测完整流程
python autoconfig/main.py

# 仅运行特征提取示例
python autoconfig/examples/feature_extraction_example.py

# 保存训练好的模型
python autoconfig/main.py --save-model model.pkl
```

### 代码示例

```python
from autoconfig import CostPredictor, FeatureManager
import networkx as nx

# 1. 创建预测器
predictor = CostPredictor()

# 2. 准备训练数据
queries = [...]  # 查询代码/图列表
graphs = [...]   # 数据图列表
configs = [...]  # 配置列表
times = [...]    # 实际执行时间

# 3. 训练模型
metrics = predictor.train(queries, graphs, configs, times, verbose=True)

# 4. 预测新查询
query_code = """
for v in G.vertices():
    for neighbor in G.neighbors(v):
        process(v, neighbor)
"""

graph = nx.erdos_renyi_graph(1000, 0.05)
config = {'memory_limit': 8192, 'num_threads': 4, ...}

predicted_time = predictor.predict(query_code, graph, config)
print(f"预计执行时间：{predicted_time:.2f} ms")
```

## 文档

| 文档 | 说明 |
|------|------|
| [快速开始](docs/quickstart.md) | 5 分钟入门指南 |
| [使用指南](docs/usage_guide.md) | 详细用法和 API 示例 |
| [API 参考](docs/api_reference.md) | 完整类和函数文档 |
| [特征提取](docs/feature_extraction.md) | 三阶段特征提取详解 |

## 项目结构

```
autoconfig/
├── autoconfig/              # 主包
│   ├── __init__.py
│   ├── main.py              # 训练和预测入口
│   ├── examples/            # 示例代码
│   ├── feature_extractor/   # 特征提取模块
│   │   ├── static_extractor.py      # 静态特征
│   │   ├── symbolic_extractor.py    # 符号特征
│   │   ├── graph_partition_extractor.py  # 图/分区特征
│   │   ├── config_extractor.py      # 配置特征
│   │   └── feature_manager.py       # 特征管理
│   ├── models/              # 模型模块
│   │   └── bayesian_model.py    # 贝叶斯模型
│   └── prediction/          # 预测模块
│       └── cost_predictor.py    # 代价预测器
├── tests/                   # 单元测试
├── docs/                    # 文档
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 特征详细说明

### 静态特征 (Static Features)

从查询源代码提取，与输入图无关：

| 特征 | 识别模式 | 性能相关性 |
|------|---------|-----------|
| 循环计数 | for/while 结构 | 重复工作区域 |
| 最大循环深度 | 嵌套循环 | 嵌套工作增长 |
| 分支计数 | if/switch 条件 | 控制流不规则性 |
| 变量计数 | 变量定义 | 局部状态大小 |
| 递归计数 | 递归调用 | 搜索扩展 |
| 原子操作计数 | atomicAdd/CAS | 并行竞争风险 |
| 同步计数 | barrier/lock | 协调开销 |
| 显式并行标志 | parallel  pragma | 并行开销 |

### 符号特征 (Symbolic Features)

识别性能关键模式，用图统计实例化：

| 模板 | 代码模式 | 实例化 | 性能影响 |
|------|---------|--------|---------|
| VScan | 顶点遍历 | \|V\| | 顶点线性工作 |
| EScan | 边遍历 | \|E\| | 边遍历工作 |
| FScan | worklist 循环 | D×\|E\|/D | 轮次敏感传播 |
| RExp | 递归扩展 | avg_degree^D | 分支搜索增长 |
| Atom | 原子更新 | \|E\|×skew | 竞争序列化 |
| Comm | 跨分区通信 | ∑deg_∂(v) | 跨分区开销 |

### 图/分区特征

| 类别 | 特征 |
|------|------|
| 图统计 | \|V\|, \|E\|, 直径，平均度，最大度，偏斜，聚类系数 |
| 分区统计 | 分区数，边界顶点，边切割比，平衡度 |

### 配置特征

| 类别 | 参数 |
|------|------|
| 内存 | memory_limit, cache_size |
| 并行 | num_threads, num_workers |
| I/O | batch_size, io_buffer_size |
| 优化 | enable_index, index_type, compression |

## 评估指标

- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数
- **Calibration**: 置信区间校准度

## 支持的语言

特征提取支持多种编程语言和伪代码：

- C/C++
- Python
- Java/Scala
- 伪代码格式

## 参考资料

- 特征提取方法基于 Hybrid 图查询成本估计研究
- 贝叶斯模型参考数据库调优工作 (Bayesian Optimization, DBTune 等)

## 许可证

MIT License
