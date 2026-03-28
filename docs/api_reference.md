# AutoConfig API 参考

## 模块概览

| 模块 | 说明 |
|------|------|
| `autoconfig` | 主包，导出所有公共类 |
| `autoconfig.feature_extractor` | 特征提取相关类 |
| `autoconfig.models` | 模型实现 |
| `autoconfig.prediction` | 预测器类 |

---

## 类 API

### CostPredictor

**位置**: `autoconfig.prediction.cost_predictor`

图查询执行时间预测器。

#### `__init__(model=None)`

```python
from autoconfig import CostPredictor, BayesianExecutionTimeModel

# 使用默认模型
predictor = CostPredictor()

# 使用自定义模型
custom_model = BayesianExecutionTimeModel(n_iter=500)
predictor = CostPredictor(model=custom_model)
```

#### `train(queries, graphs, configs, execution_times, verbose=False)`

训练预测模型。

**参数**:
- `queries`: List[Graph | Dict] - 查询图列表
- `graphs`: List[Graph | Dict] - 数据图列表
- `configs`: List[Dict] - 配置字典列表
- `execution_times`: ndarray - 实际执行时间
- `verbose`: bool - 是否打印训练进度

**返回**: Dict - 训练指标

```python
metrics = predictor.train(
    queries=train_queries,
    graphs=train_graphs,
    configs=train_configs,
    execution_times=train_times,
    verbose=True
)
# {'mae': 10.5, 'rmse': 15.2, 'mape': 5.3, 'r2': 0.92}
```

#### `predict(query, graph, config, return_uncertainty=False)`

预测单个查询的执行时间。

**参数**:
- `query`: Graph | Dict - 查询图
- `graph`: Graph | Dict - 数据图
- `config`: Dict - 配置字典
- `return_uncertainty`: bool - 是否返回置信区间

**返回**: float | Tuple[float, float, float]

```python
# 点预测
time = predictor.predict(query, graph, config)

# 区间预测
pred, lower, upper = predictor.predict(
    query, graph, config, return_uncertainty=True
)
```

#### `predict_batch(queries, graphs, configs, return_uncertainty=False)`

批量预测。

**参数**:
- `queries`: List[Graph | Dict]
- `graphs`: List[Graph | Dict]
- `configs`: List[Dict]
- `return_uncertainty`: bool

**返回**: ndarray | Tuple[ndarray, ndarray, ndarray]

```python
predictions = predictor.predict_batch(
    test_queries, test_graphs, test_configs
)
```

#### `evaluate(queries, graphs, configs, execution_times)`

评估模型性能。

**参数**:
- `queries`: List[Graph | Dict] - 测试查询
- `graphs`: List[Graph | Dict] - 测试图
- `configs`: List[Dict] - 测试配置
- `execution_times`: ndarray - 实际时间

**返回**: Dict - 评估指标

```python
metrics = predictor.evaluate(
    test_queries, test_graphs, test_configs, test_times
)
# {'mae': ..., 'rmse': ..., 'mape': ..., 'r2': ..., 'calibration': ...}
```

#### `save_model(filepath)` / `load_model(filepath)`

```python
predictor.save_model('model.pkl')
predictor.load_model('model.pkl')
```

#### `get_feature_importance()`

获取特征重要性（基于模型权重绝对值）。

**返回**: ndarray

```python
importance = predictor.get_feature_importance()
```

#### `get_feature_names()`

获取所有特征名称。

**返回**: List[str]

```python
names = predictor.get_feature_names()
```

---

### FeatureManager

**位置**: `autoconfig.feature_extractor.feature_manager`

特征提取管理器。

#### `__init__()`

```python
from autoconfig import FeatureManager

manager = FeatureManager()
```

#### `extract_all(query_graph, data_graph, config)`

从 NetworkX 图提取组合特征。

**参数**:
- `query_graph`: nx.Graph - 查询图
- `data_graph`: nx.Graph - 数据图
- `config`: Dict - 配置字典

**返回**: ndarray - 组合特征向量

```python
features = manager.extract_all(query, graph, config)
```

#### `extract_all_from_dict(query_dict, graph_dict, config)`

从字典表示提取特征。

**参数**:
- `query_dict`: Dict - `{'nodes': [...], 'edges': [...]}`
- `graph_dict`: Dict - `{'nodes': [...], 'edges': [...]}`
- `config`: Dict - 配置字典

**返回**: ndarray

```python
features = manager.extract_all_from_dict(
    {'nodes': [0,1,2], 'edges': [(0,1), (1,2)]},
    {'nodes': list(range(100)), 'edges': [...]},
    config
)
```

#### `get_feature_names()`

获取所有特征名称列表。

**返回**: List[str]

```python
names = manager.get_feature_names()
# ['query_num_nodes', 'query_num_edges', ..., 'conf_compression_enabled']
```

#### `get_feature_dimensions()`

获取各模块特征维度。

**返回**: Tuple[int, int, int] - (query_dim, graph_dim, config_dim)

```python
q_dim, g_dim, c_dim = manager.get_feature_dimensions()
# (13, 15, 10)
```

#### `normalize_features(features, means=None, stds=None)`

Z-score 标准化。

**参数**:
- `features`: ndarray - 特征矩阵
- `means`: ndarray - 预计算均值
- `stds`: ndarray - 预计算标准差

**返回**: Tuple[ndarray, ndarray, ndarray] - (normalized, means, stds)

```python
normalized, means, stds = manager.normalize_features(features)
```

---

### QueryFeatureExtractor

**位置**: `autoconfig.feature_extractor.query_extractor`

查询图特征提取器。

#### `extract(query_graph)`

从 NetworkX 图提取特征。

**返回**: ndarray (13 维)

特征列表:
- `query_num_nodes`: 节点数
- `query_num_edges`: 边数
- `query_density`: 密度
- `query_avg_degree`: 平均度
- `query_max_degree`: 最大度
- `query_min_degree`: 最小度
- `query_std_degree`: 度标准差
- `query_num_triangles`: 三角形数量
- `query_clustering_coeff`: 聚类系数
- `query_diameter`: 直径
- `query_is_connected`: 是否连通
- `query_num_components`: 连通分量数
- `query_avg_path_length`: 平均路径长度

#### `extract_from_dict(query_dict)`

从字典提取特征。

```python
extractor = QueryFeatureExtractor()
features = extractor.extract_from_dict({
    'nodes': [0, 1, 2, 3],
    'edges': [(0, 1), (1, 2), (2, 3), (0, 3)]
})
```

#### `get_feature_names()`

获取特征名称列表。

---

### GraphFeatureExtractor

**位置**: `autoconfig.feature_extractor.graph_extractor`

数据图特征提取器。

#### `extract(data_graph)`

从 NetworkX 图提取特征。

**返回**: ndarray (15 维)

特征列表:
- `graph_num_nodes`: 节点数
- `graph_num_edges`: 边数
- `graph_density`: 密度
- `graph_avg_degree`: 平均度
- `graph_max_degree`: 最大度
- `graph_min_degree`: 最小度
- `graph_std_degree`: 度标准差
- `graph_num_triangles`: 三角形数量
- `graph_clustering_coeff`: 聚类系数
- `graph_diameter`: 直径
- `graph_is_connected`: 是否连通
- `graph_num_components`: 连通分量数
- `graph_avg_path_length`: 平均路径长度
- `graph_assortativity`: 同配系数
- `graph_transitivity`: 传递性

#### `extract_from_dict(graph_dict)`

从字典提取特征。

#### `get_feature_names()`

获取特征名称列表。

---

### ConfigFeatureExtractor

**位置**: `autoconfig.feature_extractor.config_extractor`

系统配置特征提取器。

#### `extract(config)`

从配置字典提取特征。

**返回**: ndarray (10 维)

特征列表:
- `conf_memory_limit`: 内存限制 (MB)
- `conf_num_threads`: 线程数
- `conf_cache_size`: 缓存大小 (MB)
- `conf_batch_size`: 批大小
- `conf_io_buffer_size`: I/O 缓冲大小 (KB)
- `conf_num_workers`: 工作进程数
- `conf_timeout`: 超时时间 (秒)
- `conf_enable_index`: 索引启用 (0/1)
- `conf_index_type`: 索引类型编码
- `conf_compression_enabled`: 压缩启用 (0/1)

```python
extractor = ConfigFeatureExtractor()
features = extractor.extract({
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
})
```

#### `get_feature_names()`

获取特征名称列表。

---

### BayesianExecutionTimeModel

**位置**: `autoconfig.models.bayesian_model`

贝叶斯岭回归模型。

#### `__init__(alpha_1, alpha_2, lambda_1, lambda_2, n_iter, tol, use_log_transform)`

**参数**:
- `alpha_1`: float - 噪声精度先验参数 1
- `alpha_2`: float - 噪声精度先验参数 2
- `lambda_1`: float - 权重精度先验参数 1
- `lambda_2`: float - 权重精度先验参数 2
- `n_iter`: int - 最大迭代次数 (默认 300)
- `tol`: float - 收敛容差 (默认 1e-3)
- `use_log_transform`: bool - 是否使用对数变换 (默认 True)

```python
from autoconfig import BayesianExecutionTimeModel

model = BayesianExecutionTimeModel(
    alpha_1=1e-6,
    alpha_2=1e-6,
    lambda_1=1e-6,
    lambda_2=1e-6,
    n_iter=500,
    tol=1e-4,
    use_log_transform=True
)
```

#### `fit(X, y, verbose=False)`

训练模型。

**参数**:
- `X`: ndarray (n_samples, n_features) - 特征矩阵
- `y`: ndarray (n_samples,) - 目标值（执行时间）
- `verbose`: bool - 打印训练进度

**返回**: self

```python
model.fit(X_train, y_train, verbose=True)
# Iteration 0: weight_change=0.447446
# Converged at iteration 25
```

#### `predict(X, return_std=False)`

预测执行时间。

**参数**:
- `X`: ndarray (n_samples, n_features)
- `return_std`: bool - 是否返回标准差

**返回**: ndarray | Tuple[ndarray, ndarray]

```python
predictions = model.predict(X_test)
pred, std = model.predict(X_test, return_std=True)
```

#### `predict_with_uncertainty(X, confidence_level=0.95)`

预测并返回置信区间。

**参数**:
- `X`: ndarray (n_samples, n_features)
- `confidence_level`: float - 置信水平 (默认 0.95)

**返回**: Tuple[ndarray, ndarray, ndarray] - (pred, lower, upper)

```python
pred, lower, upper = model.predict_with_uncertainty(X_test, confidence_level=0.95)
```

#### `score(X, y)`

计算 R² 分数。

**返回**: float

```python
r2 = model.score(X_test, y_test)
```

#### `get_params()` / `set_params(params)`

获取/设置模型参数。

```python
params = model.get_params()
# {'weights': ..., 'sigma': ..., 'alpha': ..., 'lambda': ..., ...}

new_model = BayesianExecutionTimeModel()
new_model.set_params(params)
```

#### `save(filepath)` / `load(filepath)`

```python
model.save('model.pkl')
loaded_model = BayesianExecutionTimeModel.load('model.pkl')
```

---

## 数据结构

### 查询图表示

```python
# NetworkX Graph
import networkx as nx
query = nx.Graph()
query.add_edges_from([(0, 1), (1, 2), (2, 3)])

# 或字典
query_dict = {
    'nodes': [0, 1, 2, 3],
    'edges': [(0, 1), (1, 2), (2, 3)]
}
```

### 配置字典

```python
config = {
    'memory_limit': int,        # MB
    'num_threads': int,
    'cache_size': int,          # MB
    'batch_size': int,
    'io_buffer_size': int,      # KB
    'num_workers': int,
    'timeout': int,             # seconds
    'enable_index': bool,
    'index_type': str,          # 'none' | 'btree' | 'hash' | 'bitmap'
    'compression_enabled': bool
}
```

---

## 异常处理

```python
from autoconfig import CostPredictor

predictor = CostPredictor()

try:
    # 未训练时预测
    predictor.predict(query, graph, config)
except ValueError as e:
    print(f"Error: {e}")

try:
    # 加载损坏的模型
    predictor.load_model('corrupted_model.pkl')
except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"Error loading model: {e}")
```
