# 配置推荐功能更新说明

## 更新内容

### 1. 支持查询文件作为输入（不再使用枚举类型）

**之前**：
```bash
autoconfig recommend --query bfs --graph data/graph.csv
```
只能使用预定义的查询类型（bfs, dfs, pagerank 等）。

**现在**：
```bash
autoconfig recommend --query my_query.cu --graph data/graph.csv
```
支持任意 `.cu`、`.cpp`、`.py` 文件，系统会自动分析代码复杂度。

### 2. 修复 Top-3 配置重复问题

**之前的问题**：
```
[Rank 1] default_5_pert_+2  CPU: 136 cores, Cost: 0.5373
[Rank 2] default_5_pert_+2  CPU: 136 cores, Cost: 0.5373  # 重复！
[Rank 3] default_5_pert_+2  CPU: 136 cores, Cost: 0.5373  # 重复！
```

**现在的输出**：
```
[Rank 1] default_5_pert_+2  CPU: 136 cores, Cost: 0.5425
[Rank 2] default_5_pert_+1  CPU: 132 cores, Cost: 0.5449  # 不同配置
[Rank 3] default_5         CPU: 128 cores, Cost: 0.5473  # 不同配置
```

Top-3 现在是**3 个不同的资源配置**，按成本排序。

## 新增文件

### 1. QueryComplexityExtractor (`autoconfig/utils/query_complexity_extractor.py`)

从 CUDA/C++/Python 源代码文件中提取查询复杂度特征。

**支持的代码模式**：
- **v_scan**: 顶点扫描 (`forAllVertices`, `parallel_for(0, numVertices)`)
- **e_scan**: 边扫描 (`forAllEdges`, `parallel_for(0, numEdges)`)
- **f_scan**: 前沿/邻居扫描 (`forNeighbors`, `forFrontier`, `g.neighbors()`)
- **atomic**: 原子操作 (`atomicAdd`, `atomicMin`, `fetch_add`)
- **sync**: 同步屏障 (`barrier`, `__syncthreads`, `MPI_Barrier`)

**使用示例**：
```python
from autoconfig.utils.query_complexity_extractor import QueryComplexityExtractor

extractor = QueryComplexityExtractor()
complexity = extractor.extract_from_file('my_query.cu')
# 输出：{'v_scan': 1, 'e_scan': 0, 'f_scan': 1, 'atomic': 1, 'sync': 0}
```

### 2. 示例查询文件 (`examples/query_bfs.cu`)

提供了 BFS 查询的示例实现，包含 CPU 和 CUDA 两个版本。

## 修改的文件

### 1. `autoconfig/cli.py`

**变更**：
- `--query` 参数从 `choices=[...]` 改为接受文件路径
- 使用 `QueryComplexityExtractor` 从文件提取复杂度
- 调用新的 `recommend_with_complexity` 方法

### 2. `autoconfig/online/optimizer.py`

**核心改进**：

1. **`_config_signature()`**: 生成配置唯一标识，用于去重
2. **`generate_perturbations()`**: 生成多个不同的扰动配置（确保 CPU 核心数不同）
3. **`optimize()`**: 使用 `diverse_top_k` 逻辑，返回不同的配置

**关键代码**：
```python
def optimize(...):
    # ... 生成扰动 ...
    
    # 选择多样化的 Top-K
    diverse_top_k = []
    seen_signatures = set()
    
    for item in all_explored:
        sig = self._config_signature(item['config']['resource'])
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            diverse_top_k.append(item)
            if len(diverse_top_k) >= self.top_k:
                break
```

### 3. `autoconfig/online/recommender.py`

**新增方法**：
```python
def recommend_with_complexity(
    query_complexity: Dict[str, int],
    graph_features: Dict[str, Any],
    candidate_configs: Optional[List[Dict[str, Any]]] = None,
    top_n: int = 3
) -> Dict[str, Any]:
    """使用查询复杂度直接推荐（不依赖预定义查询名）"""
```

## 使用指南

### 基本用法

```bash
# 推荐配置
autoconfig recommend \
    --query my_algorithm.cu \
    --graph data/edges.csv \
    --top-n 3 \
    --output out/recommendation.yaml
```

### Python API

```python
from autoconfig import Recommender
from autoconfig.utils.query_complexity_extractor import QueryComplexityExtractor

# 1. 提取查询复杂度
extractor = QueryComplexityExtractor()
query_complexity = extractor.extract_from_file('my_query.cu')

# 2. 准备图特征
graph_features = {
    'num_vertices': 10000,
    'num_edges': 50000,
    'avg_degree': 5.0,
    'max_degree': 100,
    'density': 0.001,
    'avg_clustering': 0.5
}

# 3. 获取推荐
recommender = Recommender(model_dir='data/models/')
result = recommender.recommend_with_complexity(
    query_complexity=query_complexity,
    graph_features=graph_features,
    top_n=3
)

# 4. 查看结果
for rec in result['recommendations']:
    print(f"Rank {rec['rank']}: CPU={rec['resource']['cpu_cores']}, "
          f"Cost={rec['predicted_cost']:.4f}")
```

## 输出示例

```
============================================================
Top 3 Diverse Recommendations:
============================================================

[Rank 1] default_5_pert_+2
  CPU: 136 cores
  Memory: 512 GB
  GPU: 8
  Predicted Time: 0.46 ms
  Predicted Cost: 0.5425
  * Refined via perturbation

[Rank 2] default_5_pert_+1
  CPU: 132 cores
  Memory: 512 GB
  GPU: 8
  Predicted Time: 0.46 ms
  Predicted Cost: 0.5449
  * Refined via perturbation

[Rank 3] default_5
  CPU: 128 cores
  Memory: 512 GB
  GPU: 8
  Predicted Time: 0.47 ms
  Predicted Cost: 0.5473
```

## 技术细节

### 配置去重逻辑

使用配置签名（signature）来确保 Top-K 配置的唯一性：

```python
def _config_signature(self, resource: Dict[str, Any]) -> str:
    return f"CPU:{resource.get('cpu_cores', 0)}_MEM:{resource.get('memory_gb', 0)}_GPU:{resource.get('num_gpus', 0)}"
```

### 扰动生成

每个基础配置生成多个扰动版本，确保 CPU 核心数唯一：

```python
def generate_perturbations(...):
    seen_cores = {base_cpu_cores}
    
    for _ in range(num_perturbations):
        # 寻找未使用的 delta
        delta = find_unique_delta(seen_cores)
        new_cores = base_cores + delta * 4
        seen_cores.add(new_cores)
        
        # 生成扰动配置
        perturbed = perturb_config(base_config, delta)
```

## 向后兼容

旧的 `recommend()` 方法仍然可用，支持使用预定义查询名：

```python
recommender.recommend(
    query_name='bfs',  # 使用预定义类型
    graph_features=graph_features,
    top_n=3
)
```

系统会自动转换为 `recommend_with_complexity()` 调用。
