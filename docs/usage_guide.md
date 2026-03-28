# AutoConfig 使用指南

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [使用示例](#使用示例)
- [命令行工具](#命令行工具)
- [模型保存与加载](#模型保存与加载)
- [常见问题](#常见问题)

---

## 安装

### 1. 创建虚拟环境（推荐）

```bash
cd autoconfig
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装为可编辑包

```bash
pip install -e .
```

---

## 快速开始

### 30 秒入门

```python
from autoconfig import CostPredictor
import networkx as nx

# 1. 创建预测器
predictor = CostPredictor()

# 2. 准备数据
query = nx.Graph()
query.add_edges_from([(0, 1), (1, 2), (2, 3)])

graph = nx.erdos_renyi_graph(100, 0.1)

config = {
    'memory_limit': 8192,
    'num_threads': 4,
    'cache_size': 1024,
    'batch_size': 1000,
    'io_buffer_size': 64,
    'num_workers': 2,
    'timeout': 300,
    'enable_index': True,
    'index_type': 'btree',
    'compression_enabled': False
}

# 3. 预测（需要先用真实数据训练）
# predicted_time = predictor.predict(query, graph, config)
```

### 完整流程

```python
from autoconfig import CostPredictor
import networkx as nx
import numpy as np

# 1. 准备训练数据
queries = []
graphs = []
configs = []
exec_times = []

for i in range(100):
    # 生成查询图
    query = nx.erdos_renyi_graph(np.random.randint(5, 15), 0.2)
    
    # 生成数据图
    graph = nx.erdos_renyi_graph(np.random.randint(50, 200), 0.1)
    
    # 生成配置
    config = {
        'memory_limit': np.random.choice([4096, 8192, 16384]),
        'num_threads': np.random.choice([2, 4, 8, 16]),
        'cache_size': np.random.choice([512, 1024, 2048]),
        'batch_size': np.random.choice([500, 1000, 2000]),
        'io_buffer_size': np.random.choice([32, 64, 128]),
        'num_workers': np.random.choice([1, 2, 4]),
        'timeout': 300,
        'enable_index': np.random.choice([True, False]),
        'index_type': np.random.choice(['btree', 'hash', 'bitmap']),
        'compression_enabled': np.random.choice([True, False])
    }
    
    # 实际执行时间（从真实系统测量）
    actual_time = measure_execution_time(query, graph, config)
    
    queries.append(query)
    graphs.append(graph)
    configs.append(config)
    exec_times.append(actual_time)

# 2. 创建并训练预测器
predictor = CostPredictor()
metrics = predictor.train(queries, graphs, configs, exec_times, verbose=True)

# 3. 评估模型
test_metrics = predictor.evaluate(test_queries, test_graphs, test_configs, test_times)

# 4. 进行预测
new_query = nx.erdos_renyi_graph(10, 0.2)
new_graph = nx.erdos_renyi_graph(100, 0.1)
new_config = {'memory_limit': 8192, 'num_threads': 4, ...}

predicted_time = predictor.predict(new_query, new_graph, new_config)
```

---

## API 参考

### CostPredictor

主预测器类，整合特征提取和模型预测。

#### 初始化

```python
from autoconfig import CostPredictor

predictor = CostPredictor(model=None)  # 可使用预训练模型
```

#### 方法

**`train(queries, graphs, configs, execution_times, verbose=False)`**

训练模型。

```python
metrics = predictor.train(
    queries=train_queries,
    graphs=train_graphs,
    configs=train_configs,
    execution_times=train_times,
    verbose=True
)
```

返回指标字典：
- `mae`: 平均绝对误差
- `rmse`: 均方根误差
- `mape`: 平均绝对百分比误差
- `r2`: 决定系数

**`predict(query, graph, config, return_uncertainty=False)`**

预测单个查询的执行时间。

```python
# 简单预测
time = predictor.predict(query, graph, config)

# 带不确定性估计
pred, lower, upper = predictor.predict(
    query, graph, config, return_uncertainty=True
)
```

**`predict_batch(queries, graphs, configs, return_uncertainty=False)`**

批量预测。

```python
predictions = predictor.predict_batch(
    queries, graphs, configs, return_uncertainty=True
)
```

**`evaluate(queries, graphs, configs, execution_times)`**

评估模型性能。

```python
metrics = predictor.evaluate(
    test_queries, test_graphs, test_configs, test_times
)
```

**`save_model(filepath)` / `load_model(filepath)`**

保存/加载模型。

```python
predictor.save_model('model.pkl')
predictor.load_model('model.pkl')
```

**`get_feature_importance()`**

获取特征重要性。

```python
importance = predictor.get_feature_importance()
names = predictor.get_feature_names()
```

---

### FeatureManager

特征管理器，处理 Q、G、Conf 的特征提取。

```python
from autoconfig import FeatureManager

manager = FeatureManager()

# 从 NetworkX 图提取
features = manager.extract_all(query_graph, data_graph, config)

# 从字典提取
features = manager.extract_all_from_dict(query_dict, graph_dict, config)

# 获取特征名称
names = manager.get_feature_names()

# 获取特征维度
query_dim, graph_dim, config_dim = manager.get_feature_dimensions()
```

---

### BayesianExecutionTimeModel

贝叶斯执行时间预测模型。

```python
from autoconfig import BayesianExecutionTimeModel

model = BayesianExecutionTimeModel(
    alpha_1=1e-6,      # 噪声精度先验参数
    alpha_2=1e-6,
    lambda_1=1e-6,     # 权重精度先验参数
    lambda_2=1e-6,
    n_iter=300,        # 最大迭代次数
    tol=1e-3,          # 收敛容差
    use_log_transform=True  # 对数变换
)

# 训练
model.fit(X_train, y_train, verbose=True)

# 预测
predictions = model.predict(X_test)

# 带不确定性预测
pred, lower, upper = model.predict_with_uncertainty(X_test, confidence_level=0.95)

# 评估
r2 = model.score(X_test, y_test)

# 保存/加载
model.save('model.pkl')
loaded_model = BayesianExecutionTimeModel.load('model.pkl')
```

---

## 使用示例

### 示例 1：基本预测流程

```python
from autoconfig import CostPredictor
import networkx as nx

# 创建预测器
predictor = CostPredictor()

# 假设已有训练数据
# predictor.train(queries, graphs, configs, times)

# 创建查询
query = nx.Graph()
query.add_edges_from([
    (0, 1), (1, 2), (2, 3),  # 路径
    (0, 3)                    # 形成环
])

# 加载或创建数据图
graph = nx.readwrite.gml.read_gml('data_graph.gml')

# 配置系统参数
config = {
    'memory_limit': 16384,    # 16GB
    'num_threads': 8,
    'cache_size': 2048,       # 2GB
    'batch_size': 2000,
    'io_buffer_size': 128,    # 128KB
    'num_workers': 4,
    'timeout': 600,
    'enable_index': True,
    'index_type': 'hash',
    'compression_enabled': False
}

# 预测
predicted_time = predictor.predict(query, graph, config)
print(f"预计执行时间：{predicted_time:.2f} ms")
```

### 示例 2：配置优化

```python
import numpy as np
from itertools import product

# 定义配置搜索空间
config_space = {
    'num_threads': [2, 4, 8, 16],
    'memory_limit': [4096, 8192, 16384],
    'cache_size': [512, 1024, 2048]
}

# 网格搜索最优配置
best_time = float('inf')
best_config = None

for threads, memory, cache in product(
    config_space['num_threads'],
    config_space['memory_limit'],
    config_space['cache_size']
):
    config = {
        'num_threads': threads,
        'memory_limit': memory,
        'cache_size': cache,
        'batch_size': 1000,
        'io_buffer_size': 64,
        'num_workers': 2,
        'timeout': 300,
        'enable_index': True,
        'index_type': 'btree',
        'compression_enabled': False
    }
    
    pred_time = predictor.predict(query, graph, config)
    
    if pred_time < best_time:
        best_time = pred_time
        best_config = config

print(f"最优配置：{best_config}")
print(f"预计时间：{best_time:.2f} ms")
```

### 示例 3：不确定性分析

```python
# 获取预测分布
pred, lower, upper = predictor.predict(
    query, graph, config, return_uncertainty=True
)

print(f"预测值：{pred:.2f} ms")
print(f"95% 置信区间：[{lower:.2f}, {upper:.2f}] ms")
print(f"不确定性：{(upper - lower) / pred * 100:.1f}%")

# 高风险预测（不确定性高）需要谨慎
if (upper - lower) / pred > 0.5:
    print("警告：预测不确定性较高，建议实际测量验证")
```

### 示例 4：特征分析

```python
# 获取特征重要性
importance = predictor.get_feature_importance()
names = predictor.get_feature_names()

# 排序并显示最重要的特征
sorted_idx = np.argsort(importance)[::-1]
print("Top 10 重要特征:")
for i in sorted_idx[:10]:
    print(f"  {names[i]}: {importance[i]:.4f}")
```

---

## 命令行工具

### 训练模型

```bash
# 基本训练（80 训练样本 + 20 测试样本）
python autoconfig/main.py --train

# 自定义样本数量
python autoconfig/main.py --train --n-train 200 --n-test 50

# 保存模型
python autoconfig/main.py --train --save-model my_model.pkl
```

### 运行预测

```bash
# 运行预测示例
python autoconfig/main.py --predict
```

### 完整流程

```bash
# 训练 + 预测 + 保存
python autoconfig/main.py --train --n-train 100 --n-test 20 --save-model final_model.pkl
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train` | 运行训练流程 | - |
| `--predict` | 运行预测示例 | - |
| `--n-train` | 训练样本数 | 80 |
| `--n-test` | 测试样本数 | 20 |
| `--save-model` | 模型保存路径 | None |

---

## 模型保存与加载

### 保存训练好的模型

```python
# 方法 1：通过 CostPredictor
predictor.save_model('models/execution_time_model.pkl')

# 方法 2：直接保存 Bayesian 模型
predictor.model.save('models/bayesian_model.pkl')
```

### 加载模型

```python
from autoconfig import CostPredictor, BayesianExecutionTimeModel

# 方法 1：加载到 CostPredictor
predictor = CostPredictor()
predictor.load_model('models/execution_time_model.pkl')

# 方法 2：直接加载 Bayesian 模型
model = BayesianExecutionTimeModel.load('models/bayesian_model.pkl')
predictor = CostPredictor(model=model)
```

### 部署使用

```python
# deploy.py - 部署脚本
from autoconfig import CostPredictor

# 加载预训练模型
predictor = CostPredictor()
predictor.load_model('production_model.pkl')

# 在生产环境中使用
def estimate_query_time(query, graph, config):
    return predictor.predict(query, graph, config)
```

---

## 常见问题

### Q1: 训练数据从哪里来？

训练数据需要从实际系统中收集：

```python
import time

def measure_execution_time(query, graph, config):
    """在实际系统中执行查询并记录时间"""
    start = time.time()
    
    # 在实际图数据库/系统中执行查询
    # result = execute_query(query, graph, config)
    
    end = time.time()
    return (end - start) * 1000  # 转换为毫秒
```

### Q2: 需要多少训练数据？

- 最少：30-50 个样本（基础模型）
- 推荐：100-200 个样本（稳定性能）
- 理想：500+ 个样本（高精度）

### Q3: 如何处理新的图类型？

如果新图与训练数据差异较大：

1. 收集新图类型的样本
2. 增量训练或重新训练
3. 使用不确定性估计识别分布外样本

```python
pred, lower, upper = predictor.predict(query, graph, config, return_uncertainty=True)

# 检查不确定性
if (upper - lower) / pred > 0.5:
    print("警告：该查询可能超出训练分布")
```

### Q4: 预测精度如何提升？

1. **增加训练数据量**
2. **改进特征工程** - 添加领域特定特征
3. **调整模型超参数**
4. **使用集成方法**

```python
# 自定义贝叶斯模型参数
from autoconfig import BayesianExecutionTimeModel, CostPredictor

model = BayesianExecutionTimeModel(
    n_iter=500,      # 增加迭代次数
    tol=1e-4,        # 更严格的收敛条件
    use_log_transform=True
)

predictor = CostPredictor(model=model)
```

### Q5: 支持哪些图格式？

支持 NetworkX 的所有图格式：

```python
import networkx as nx

# 从文件加载
G = nx.readwrite.gml.read_gml('graph.gml')
G = nx.readwrite.graphml.read_graphml('graph.graphml')
G = nx.readwrite.adjlist.read_adjlist('graph.adjlist')

# 从边列表创建
edges = [(0, 1), (1, 2), (2, 3)]
G = nx.Graph(edges)

# 作为字典传递
graph_dict = {'nodes': [0, 1, 2], 'edges': [(0, 1), (1, 2)]}
```

---

## 联系与支持

如有问题或建议，请查阅：
- [README.md](../README.md) - 项目概述
- [源代码](../autoconfig/) - 实现细节
