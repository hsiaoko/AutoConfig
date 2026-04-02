# AutoConfig 快速使用指南

## 安装

```bash
cd /path/to/AutoConfig
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## 架构概览

AutoConfig 分为三个主要模块：

1. **特征提取模块** (`feature_extractor/`) - 提取查询、图、配置的特征
2. **Offline 训练模块** (`offline/`) - 生成数据、训练模型
3. **Online 推荐模块** (`online/`) - 预测代价、推荐最优配置

## 使用流程

### 步骤 1: 训练模型

```bash
# 方式 1: 使用 CLI（推荐）
autoconfig train --n-samples 500 --output data/models/

# 方式 2: 使用脚本
./scripts/train_pipeline.sh 500 data/models/

# 方式 3: 使用 Python API
from autoconfig import Trainer

trainer = Trainer('data/models/')
metrics = trainer.train(num_samples=500)
print(f"Time model R²: {metrics['time_model']['test_r2']}")
print(f"Cost model R²: {metrics['cost_model']['test_r2']}")
```

### 步骤 2: 生成训练数据（可选）

```bash
# 生成模拟数据
autoconfig generate-data --n-samples 200 --output data/generated/
```

### 步骤 3: 推荐配置

```bash
# 使用 CLI 推荐
autoconfig recommend \
    --query bfs \
    --graph data/graph.csv \
    --top-n 3 \
    --output out/recommendation.yaml
```

支持的查询类型：`bfs`, `dfs`, `pagerank`, `cc`, `sssp`, `kcore`, `tc`

### Python API 完整示例

```python
from autoconfig import DataGenerator, Trainer, Recommender

# 1. 生成训练数据
generator = DataGenerator(seed=42)
dataset = generator.generate_dataset(num_samples=500)
generator.save_dataset(dataset, 'data/generated/')

# 2. 训练模型
trainer = Trainer('data/models/')
metrics = trainer.train(num_samples=500)
print(f"Models trained successfully!")

# 3. 推荐配置
recommender = Recommender(model_dir='data/models/')

# 准备图特征
graph_features = {
    'num_vertices': 10000,
    'num_edges': 50000,
    'avg_degree': 5.0,
    'max_degree': 100,
    'density': 0.001,
    'avg_clustering': 0.5
}

# 获取推荐
result = recommender.recommend(
    query_name='pagerank',
    graph_features=graph_features,
    top_n=3
)

print(f"\nBest configuration:")
best = result['best_config']
print(f"  Config ID: {best['config_id']}")
print(f"  CPU: {best['resource']['cpu_cores']} cores")
print(f"  Memory: {best['resource']['memory_gb']} GB")
print(f"  GPU: {best['resource']['num_gpus']}")
print(f"  Predicted time: {best['predicted_time_ms']:.2f} ms")
print(f"  Predicted cost: {best['predicted_cost']:.4f}")
```

## CLI 命令参考

### 特征提取

```bash
# 提取查询代码特征
autoconfig query --input query.py --output out/query.yaml

# 提取图特征
autoconfig graph --input graph.csv --output out/graph.yaml

# 生成配置样本
autoconfig config --num-samples 20 --output out/configs.yaml

# 完整流程
autoconfig all --query query.py --graph data/ --config-n 20 --output out/
```

### 训练

```bash
# 训练模型
autoconfig train --n-samples 500 --output data/models/

# 生成数据
autoconfig generate-data \
    --n-samples 200 \
    --graph-min 100 \
    --graph-max 5000 \
    --output data/generated/
```

### 推荐

```bash
# 推荐最优配置
autoconfig recommend \
    --query bfs \
    --graph data/graph.csv \
    --top-n 3

# 使用特定模型目录
autoconfig recommend \
    --query pagerank \
    --graph data/graph.csv \
    --model-dir data/models/ \
    --output out/result.yaml
```

## 模块详解

### DataGenerator - 数据生成器

生成模拟的训练数据，包括：
- 7 种查询模板（BFS, DFS, PageRank, CC, SSSP, KCore, TriangleCounting）
- 多种图类型和大小
- 6 种资源配置

```python
from autoconfig import DataGenerator

dg = DataGenerator(seed=42)

# 生成单个样本
sample = dg.generate_sample(query_name='bfs')
print(f"Time: {sample['labels']['execution_time_ms']} ms")
print(f"Cost: {sample['labels']['execution_cost']}")

# 生成数据集
dataset = dg.generate_dataset(num_samples=100)
```

### Trainer - 训练器

训练两个贝叶斯模型：
- **时间预测模型**: 预测执行时间
- **代价预测模型**: 预测资源成本

```python
from autoconfig import Trainer

trainer = Trainer('data/models/')
metrics = trainer.train(num_samples=500, verbose=True)

# 查看指标
print(f"Time model R²: {metrics['time_model']['test_r2']}")
print(f"Cost model R²: {metrics['cost_model']['test_r2']}")
```

### Recommender - 推荐服务

提供配置推荐服务：

```python
from autoconfig import Recommender

recommender = Recommender(model_dir='data/models/')

# 推荐
result = recommender.recommend('bfs', graph_features, top_n=3)

# What-if 分析
analysis = recommender.what_if('bfs', graph_features, config)

# 配置对比
comparison = recommender.compare_configs('bfs', graph_features, [config1, config2, config3])
```

### ConfigurationOptimizer - 配置优化器

实现排序 + 扰动精炼：

```python
from autoconfig import CostPredictor, ConfigurationOptimizer

predictor = CostPredictor('data/models/')
optimizer = ConfigurationOptimizer(
    cost_predictor=predictor,
    top_k=3,
    perturbation_range=2,
    num_perturbations=10
)

# 优化
best = optimizer.optimize(query_complexity, graph_features, candidate_configs)
```

## 输出示例

### 训练输出

```
============================================================
Training Bayesian Models
============================================================

Generating 500 synthetic samples...
Extracting features...
Train samples: 400, Test samples: 100
Feature dimension: 16

[1/2] Training execution time model...
  Train R²: 0.9234
  Test R²: 0.8956

[2/2] Training execution cost model...
  Train R²: 0.9187
  Test R²: 0.8823

============================================================
Training Complete!
Models saved to: data/models/
============================================================
```

### 推荐输出

```
============================================================
Top 3 Recommendations for 'bfs':
============================================================

[Rank 1] config_2_pert_-1
  CPU: 12 cores
  Memory: 64 GB
  GPU: 1
  Predicted Time: 45.23 ms
  Predicted Cost: 12.5000
  * Refined via perturbation

[Rank 2] config_1
  CPU: 8 cores
  Memory: 32 GB
  GPU: 0
  Predicted Time: 67.89 ms
  Predicted Cost: 13.2000

[Rank 3] config_3
  CPU: 32 cores
  Memory: 128 GB
  GPU: 2
  Predicted Time: 23.45 ms
  Predicted Cost: 15.8000
```

## 故障排除

### 模型未找到

```
Warning: No trained models found.
Using heuristic-based recommendation.
```

解决：先运行 `autoconfig train` 训练模型。

### 导入错误

```
ModuleNotFoundError: No module named 'numpy'
```

解决：激活虚拟环境并安装依赖：
```bash
source venv/bin/activate
pip install -e .
```

## 下一步

1. 查看 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 了解详细架构
2. 查看 API 文档了解完整接口
3. 查看示例代码学习最佳实践
