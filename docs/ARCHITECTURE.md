# AutoConfig 架构文档

## 系统概述

AutoConfig 是一个基于机器学习的图查询配置推荐系统，通过分析查询模式、图数据特征和系统配置，预测执行时间和成本，并推荐最优配置。

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        AutoConfig 系统                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │   Offline 训练部分   │         │   Online 推荐部分    │       │
│  │                     │         │                     │       │
│  │  ┌───────────────┐  │         │  ┌───────────────┐  │       │
│  │  │ 数据生成器     │  │         │  │ 代价预测器    │  │       │
│  │  │ DataGenerator │  │         │  │ CostPredictor │  │       │
│  │  └───────┬───────┘  │         │  └───────┬───────┘  │       │
│  │          │          │         │          │          │       │
│  │  ┌───────▼───────┐  │         │  ┌───────▼───────┐  │       │
│  │  │ 训练器        │  │         │  │ 优化器        │  │       │
│  │  │ Trainer       │  │         │  │ Optimizer     │  │       │
│  │  └───────┬───────┘  │         │  └───────┬───────┘  │       │
│  │          │          │         │          │          │       │
│  │  ┌───────▼───────┐  │         │  ┌───────▼───────┐  │       │
│  │  │ 模型注册表    │  │         │  │ 推荐服务      │  │       │
│  │  │ ModelRegistry │──┼─────────┼─▶│ Recommender   │  │       │
│  │  └───────────────┘  │         │  └───────────────┘  │       │
│  │                     │         │                     │       │
│  │  输出：模型文件      │         │  输入：查询 + 图     │       │
│  │  - time_model.pkl   │         │  输出：最优配置     │       │
│  │  - cost_model.pkl   │         │                     │       │
│  └─────────────────────┘         └─────────────────────┘       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                        特征提取模块                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Query       │  │ Graph       │  │ Config      │            │
│  │ Extractor   │  │ Extractor   │  │ Extractor   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
autoconfig/
├── autoconfig/                    # 主包
│   ├── __init__.py                # 导出所有公共 API
│   ├── cli.py                     # 命令行接口
│   │
│   ├── feature_extractor/         # 特征提取模块
│   │   ├── __init__.py
│   │   ├── static_extractor.py    # 静态特征（循环、分支等）
│   │   ├── symbolic_extractor.py  # 符号特征（VScan, EScan 等）
│   │   ├── graph_partition_extractor.py
│   │   ├── config_extractor.py    # 配置特征
│   │   └── feature_manager.py     # 统一特征管理
│   │
│   ├── offline/                   # Offline 训练部分
│   │   ├── __init__.py
│   │   ├── data_generator.py      # 模拟数据生成器
│   │   ├── trainer.py             # 模型训练器
│   │   └── model_registry.py      # 模型注册管理
│   │
│   ├── online/                    # Online 推荐部分
│   │   ├── __init__.py
│   │   ├── cost_predictor.py      # 执行代价预测
│   │   ├── optimizer.py           # 配置优化器
│   │   └── recommender.py         # 推荐服务接口
│   │
│   └── models/                    # 模型层
│       ├── __init__.py
│       ├── bayesian_model.py      #  legacy 贝叶斯模型
│       └── bayesian_models.py     # 新版时间/代价模型
│
├── data/
│   ├── generated/                 # 生成的训练数据
│   └── models/                    # 训练好的模型
│
├── scripts/
│   ├── train_pipeline.sh          # 训练脚本
│   └── serve_recommendation.sh    # 推荐服务脚本
│
├── examples/                      # 示例数据
├── tests/                         # 单元测试
└── docs/                          # 文档
```

## 模块详解

### 1. 特征提取模块 (feature_extractor)

负责从三个维度提取特征：

**静态特征** (Static Features):
- 循环数量
- 分支数量
- 递归调用
- 原子操作
- 同步操作

**符号特征** (Symbolic Features):
- VScan: 顶点扫描系数
- EScan: 边扫描系数
- FScan: 前沿迭代系数
- Atom: 原子操作系数
- Comm: 通信系数

**图特征** (Graph Features):
- 顶点数、边数
- 度分布统计
- 密度、聚类系数
- 分区质量指标

**配置特征** (Config Features):
- CPU 核心数
- 内存大小
- GPU 数量
- 存储容量

### 2. Offline 训练模块 (offline)

#### DataGenerator - 数据生成器

生成模拟训练数据，包括：
- 7 种查询模板：BFS, DFS, PageRank, CC, SSSP, KCore, TriangleCounting
- 3 种图类型：Erdos-Renyi, PowerLaw, Community
- 6 种资源配置模板

**执行时间模型**:
```
time = (base_cost + graph_factor) * resource_factor + overhead
```

**执行成本模型**:
```
cost = (cpu_cost + mem_cost + gpu_cost) * duration
```

#### Trainer - 训练器

训练流程：
1. 加载或生成数据集
2. 提取特征向量（16 维：5 查询 + 6 图 + 5 资源）
3. 训练贝叶斯时间预测模型
4. 训练贝叶斯代价预测模型
5. 评估并保存模型

#### ModelRegistry - 模型注册表

管理模型版本和元数据：
- 版本控制
- 训练指标追踪
- 模型状态管理（active/deprecated）

### 3. Online 推荐模块 (online)

#### CostPredictor - 代价预测器

使用训练好的贝叶斯模型进行预测：
- `predict_time()`: 预测执行时间
- `predict_cost()`: 预测执行成本
- `predict_both()`: 同时预测两者
- 支持不确定性估计（置信区间）

#### ConfigurationOptimizer - 配置优化器

优化流程：
1. **排序**: 对所有候选配置按预测成本排序
2. **选择**: 选取 Top-K 配置（默认 K=3）
3. **扰动**: 对每个 Top 配置进行随机扰动（±n 个 core）
4. **精炼**: 评估扰动后的配置，选择最优

#### Recommender - 推荐服务

高层接口，提供：
- `recommend()`: 单查询推荐
- `recommend_for_graph()`: 针对具体图的推荐
- `what_if()`: What-if 分析
- `compare_configs()`: 配置对比

## 使用流程

### 训练流程

```bash
# 1. 生成训练数据（可选，train 命令会自动生成）
python -m autoconfig generate-data --n-samples 500 --output data/generated/

# 2. 训练模型
python -m autoconfig train --n-samples 500 --output data/models/

# 或使用脚本
./scripts/train_pipeline.sh 500 data/models/
```

### 推荐流程

```bash
# 推荐最优配置
python -m autoconfig recommend \
    --query bfs \
    --graph data/graph.csv \
    --top-n 3 \
    --output out/recommendation.yaml

# What-if 分析（通过 Python API）
from autoconfig import Recommender

recommender = Recommender(model_dir='data/models/')
result = recommender.what_if(
    query_name='bfs',
    graph_features={'num_vertices': 1000, 'num_edges': 5000, ...},
    config={'resource': {'cpu_cores': 16, 'memory_gb': 64, ...}}
)
```

### Python API 示例

```python
from autoconfig import DataGenerator, Trainer, Recommender

# 1. 生成数据
generator = DataGenerator(seed=42)
dataset = generator.generate_dataset(num_samples=500)
generator.save_dataset(dataset, 'data/generated/')

# 2. 训练模型
trainer = Trainer('data/models/')
metrics = trainer.train(num_samples=500)
print(f"Time model R²: {metrics['time_model']['test_r2']}")
print(f"Cost model R²: {metrics['cost_model']['test_r2']}")

# 3. 推荐配置
recommender = Recommender(model_dir='data/models/')
result = recommender.recommend(
    query_name='pagerank',
    graph_features={
        'num_vertices': 10000,
        'num_edges': 50000,
        'avg_degree': 5.0,
        'max_degree': 100,
        'density': 0.001,
        'avg_clustering': 0.5
    },
    top_n=3
)
print(f"Best config: {result['best_config']}")
```

## 数据流

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │     │    Graph    │     │   Config    │
│   Code      │     │    Data     │     │  Catalog    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────┐
│                   Feature Extraction                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Static    │  │   Symbolic  │  │    Graph    │     │
│  │  Features   │  │  Features   │  │  Features   │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         └────────────────┴────────────────┘             │
│                          │                              │
│                          ▼                              │
│              Combined Feature Vector (16-D)             │
└─────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
┌─────────────────────┐         ┌─────────────────────┐
│   Offline Training  │         │   Online Inference  │
│                     │         │                     │
│  ┌───────────────┐  │         │  ┌───────────────┐  │
│  │ DataGenerator │  │         │  │ CostPredictor │  │
│  └───────┬───────┘  │         │  └───────┬───────┘  │
│          │          │         │          │          │
│          ▼          │         │          ▼          │
│  ┌───────────────┐  │         │  ┌───────────────┐  │
│  │ Trainer       │──┼───┐     │  │ Optimizer     │  │
│  └───────┬───────┘  │   │     │  └───────┬───────┘  │
│          │          │   │     │          │          │
│          ▼          │   │     │          ▼          │
│  ┌───────────────┐  │   │     │  ┌───────────────┐  │
│  │ Model Files   │───┼───┴─────┼─▶│ Recommender   │  │
│  │ *.pkl         │  │         │  │               │  │
│  └───────────────┘  │         │  └───────────────┘  │
└─────────────────────┘         └─────────────────────┘
```

## 模型详情

### 贝叶斯时间预测模型

**输入特征** (16 维):
- 查询复杂度 (5): v_scan, e_scan, f_scan, atomic, sync
- 图统计 (6): num_vertices, num_edges, avg_degree, max_degree, density, avg_clustering
- 资源配置 (5): cpu_cores, memory_gb, num_gpus, gpu_memory_gb, storage_gb

**输出**: 预测执行时间 (ms) + 置信区间

**模型特点**:
- 对数变换目标变量（处理偏态分布）
- Z-score 特征标准化
- 变分推断求解
- 自动正则化

### 贝叶斯成本预测模型

与时间模型架构相同，但训练目标为执行成本（货币单位）。

**成本计算**:
```
cost = (cpu_cores * cpu_rate + memory * mem_rate + gpus * gpu_rate) * duration
```

## 扩展性

### 添加新的查询类型

在 `DataGenerator.QUERY_TEMPLATES` 中添加：

```python
'new_query': {
    'code': '''...''',
    'complexity': {'v_scan': 1, 'e_scan': 0, 'f_scan': 1, 'atomic': 0, 'sync': 0},
    'base_cost': 100.0,
}
```

### 添加新的资源配置

在 `DataGenerator.RESOURCE_CONFIGS` 中添加新配置元组。

### 添加新的特征

1. 在对应的 extractor 中实现特征提取逻辑
2. 更新 `_build_feature_vector()` 方法
3. 重新训练模型

## CLI 命令参考

```bash
# 特征提取
autoconfig query --input query.py --output out/query.yaml
autoconfig graph --input graph.csv --output out/graph.yaml
autoconfig config --num-samples 20 --output out/configs.yaml

# 训练
autoconfig train --n-samples 500 --output data/models/
autoconfig generate-data --n-samples 200 --output data/generated/

# 推荐
autoconfig recommend --query bfs --graph data/graph.csv --top-n 3
```

## API 参考

完整 API 文档请参阅 [docs/api_reference.md](docs/api_reference.md)
