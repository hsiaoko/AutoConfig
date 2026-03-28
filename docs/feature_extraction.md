# Hybrid 查询特征提取指南

## 概述

本系统实现了论文中描述的三阶段特征提取方法，用于 Hybrid 图查询任务的成本估计。

### 特征类型

| 类型 | 说明 | 特征数 | 来源 |
|------|------|--------|------|
| **静态特征** | 查询代码的结构分析 | 8 | 源代码 |
| **符号特征** | 工作负载模板识别 | 12 | 代码模式 + 图统计 |
| **图/分区特征** | 图和分区的统计信息 | 17 | 图 G + 分区 F |
| **配置特征** | 系统配置参数 | 10 | 配置字典 |
| **总计** | | **47** | |

---

## 特征提取流程

```
查询代码 → [静态分析] → Φ_static
         ↓
查询代码 + 图统计 → [符号匹配] → Φ_sym
         ↓
图 G + 分区 F → [统计分析] → Φ_graph
         ↓
Φ_static + Φ_sym + Φ_graph + Config → 特征向量 → 成本预测
```

---

## 静态特征 (Static Features)

从查询源代码提取的结构特征，与输入图无关。

### 特征列表

| # | 特征名 | 识别的代码模式 | 性能相关性 |
|---|--------|---------------|-----------|
| 1 | `static_loop_count` | 循环结构 (for/while) | 重复工作区域 |
| 2 | `static_max_loop_depth` | 嵌套循环 | 嵌套工作增长 |
| 3 | `static_branch_count` | 条件分支 (if/switch) | 控制流不规则性 |
| 4 | `static_variable_count` | 变量定义 | 局部状态大小 |
| 5 | `static_recursion_count` | 递归过程 | 搜索和传播扩展 |
| 6 | `static_atomic_op_count` | 原子操作 | 并行竞争风险 |
| 7 | `static_sync_count` | 同步原语 (barrier/lock) | 协调开销 |
| 8 | `static_explicit_parallel_flag` | PRAM 并行操作符 | 并行执行开销 |

### 使用示例

```python
from autoconfig.feature_extractor import StaticFeatureExtractor

extractor = StaticFeatureExtractor()

# 从源代码字符串提取
code = """
for v in G.vertices():
    if condition(v):
        process(v)
"""

features = extractor.extract(code)
print(f"Loop count: {features[0]}")
print(f"Branch count: {features[2]}")

# 从文件提取
features = extractor.extract_from_file('query.cpp')
```

### 支持的语言模式

- C/C++ 风格：`for(;;)`, `while()`, `if()`
- Python 风格：`for x in y:`, `while cond:`
- 伪代码风格：`for v in Vertices:`, `if !condition:`

---

## 符号特征 (Symbolic Features)

识别代码中的性能关键模式，创建偏函数，用图/分区统计实例化。

### 特征模板

| # | 特征模板 | 识别的代码模式 | 实例化公式 | 性能相关性 |
|---|---------|---------------|-----------|-----------|
| 1 | VScan(V, n) | 顶点列表扫描 | \|V\| 或 \|V\|/n | 顶点线性工作 |
| 2 | EScan(E, n) | 边列表扫描/邻居遍历 | \|E\| 或 \|E\|/n | 边遍历工作 |
| 3 | FScan(G, n) | Worklist/前沿驱动循环 | ∑|E_t\| (t=1 to D) | 轮次敏感传播 |
| 4 | RExp(G) | 递归邻居扩展 | ∏E[deg(v_i)] | 分支搜索增长 |
| 5 | Atom(G) | 原子写入/CAS | \|E\| × skew(G) | 竞争和序列化 |
| 6 | Comm(G, F) | 跨分区通信 | ∑deg_∂(v) | 跨分区通信 |

### 需要的图统计

| 符号特征 | 需要的图统计 |
|---------|-------------|
| VScan | \|V\| (顶点数) |
| EScan | \|E\| (边数) |
| FScan | D (直径), \|E\| |
| RExp | D (直径), avg_degree |
| Atom | \|E\|, skew (度偏斜) |
| Comm | boundary_degree_sum |

### 使用示例

```python
from autoconfig.feature_extractor import SymbolicFeatureExtractor

extractor = SymbolicFeatureExtractor()

# 准备图统计
graph_stats = {
    'num_vertices': 10000,
    'num_edges': 50000,
    'diameter': 5,
    'avg_degree': 10.0,
    'max_degree': 150,
    'skew': 3.5,
}

# BFS 代码
bfs_code = """
BFS(Graph G, source):
  worklist = [source]
  while !worklist.empty():
    for v in worklist:
      for neighbor in G.neighbors(v):
        visit(neighbor)
"""

# 提取符号特征（自动实例化）
features = extractor.extract(bfs_code, graph_stats)

# 特征解释
names = extractor.get_feature_names()
for name, value in zip(names, features):
    print(f"{name}: {value}")
```

### 输出示例

对于 BFS 查询：
```
sym_vscan_coeff: 0.0          # 无顶点扫描
sym_vscan_requires: No
sym_escan_coeff: 50000.0      # 边扫描：|E|
sym_escan_requires: Yes
sym_fscan_coeff: 50000.0      # 前沿迭代：D * |E|/D
sym_fscan_requires: Yes
sym_rexp_coeff: 0.0           # 无递归扩展
sym_atom_coeff: 0.0           # 无原子操作
sym_comm_coeff: 0.0           # 无跨分区通信
```

---

## 图和分区特征 (Graph & Partition Features)

从输入图 G 和分区 F 提取的统计信息。

### 图统计特征

| # | 特征名 | 说明 |
|---|--------|------|
| 1 | `graph_num_vertices` | 顶点数 \|V\| |
| 2 | `graph_num_edges` | 边数 \|E\| |
| 3 | `graph_diameter` | 图直径 D |
| 4 | `graph_avg_degree` | 平均度 |
| 5 | `graph_max_degree` | 最大度 |
| 6 | `graph_min_degree` | 最小度 |
| 7 | `graph_degree_std` | 度标准差 |
| 8 | `graph_skew` | 度偏斜 (max_degree / avg_degree) |
| 9 | `graph_clustering_coeff` | 聚类系数 |
| 10 | `graph_num_components` | 连通分量数 |

### 分区统计特征

| # | 特征名 | 说明 |
|---|--------|------|
| 11 | `partition_num_partitions` | 分区数 |
| 12 | `partition_boundary_vertices` | 边界顶点数 |
| 13 | `partition_boundary_degree_sum` | 边界顶点度数和 |
| 14 | `partition_avg_partition_size` | 平均分区大小 |
| 15 | `partition_size_std` | 分区大小标准差 |
| 16 | `partition_edge_cut_ratio` | 边切割比例 |
| 17 | `partition_balance` | 分区平衡度 (1.0 = 完美平衡) |

### 使用示例

```python
from autoconfig.feature_extractor import GraphPartitionExtractor
import networkx as nx

extractor = GraphPartitionExtractor()

# 创建图
graph = nx.erdos_renyi_graph(1000, 0.05)

# 创建分区
partitions = {
    0: list(range(500)),
    1: list(range(500, 1000)),
}

# 提取特征
features = extractor.extract(graph, partitions)

# 获取字典形式（用于符号特征实例化）
stats_dict = extractor.get_graph_stats_dict(graph, partitions)
print(f"num_vertices: {stats_dict['num_vertices']}")
print(f"diameter: {stats_dict['diameter']}")
print(f"skew: {stats_dict['skew']}")
```

---

## 完整流程示例

### 示例 1：子图同构查询

```python
from autoconfig.feature_extractor import FeatureManager
import networkx as nx

# 初始化特征管理器
manager = FeatureManager()

# 子图同构查询代码
subiso_code = """
PEval(Fragment F, Context ctx):
  for v in F.Vertices():
    if !ctx.IsCandidate(v, p[0]): continue
    Match m
    m.Bind(P[0], v)
    Expand(m, 1, P, F, ctx)

Expand(Match& m, int level, Pattern P, Fragment F, Context ctx):
  if level == PatternSize: return
  for v in neighbor(m.Last()):
    if !ctx.IsCandidate(v, P[level]) continue
    if !ctx.Consistent(m, v, P[level]) continue
    m.Bind(P[level], v)
    Expand(m, level + 1, P, F, ctx)  # 递归扩展
    m.Unbind(P[level])
    if ctx.IsMirror(v):  # 跨分区通信
      ctx.SendTo(context.Owner(v), ComputeMessage(v, context))
"""

# 创建输入图
graph = nx.erdos_renyi_graph(5000, 0.01)

# 创建分区
nodes = list(graph.nodes())
partitions = {
    0: nodes[:2500],
    1: nodes[2500:],
}

# 系统配置
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
    'compression_enabled': False,
}

# 提取所有特征
features = manager.extract_all(
    source_code=subiso_code,
    graph=graph,
    config=config,
    partitions=partitions
)

print(f"总特征数：{len(features)}")
print(f"特征维度：{manager.get_feature_dimensions()}")
```

### 示例 2：比较不同查询的特征

```python
# PageRank 代码
pagerank_code = """
PageRank(Graph G, int max_iter, float damping):
  for v in G.vertices():
    rank[v] = 1.0 / |V|
  
  for iter in 1..max_iter:
    for v in G.vertices():
      contribution = rank[v] / out_degree(v)
      for neighbor in G.out_edges(v):
        atomicAdd(new_rank[neighbor], contribution)
  
  barrier()
"""

# BFS 代码
bfs_code = """
BFS(Graph G, vertex source):
  worklist = [source]
  visited[source] = true
  
  while !worklist.empty():
    next_worklist = []
    for v in worklist:
      for neighbor in G.neighbors(v):
        if !visited[neighbor]:
          visited[neighbor] = true
          next_worklist.append(neighbor)
    worklist = next_worklist
"""

# 提取并比较静态特征
from autoconfig.feature_extractor import StaticFeatureExtractor

extractor = StaticFeatureExtractor()

queries = {
    'Subgraph Iso': subiso_code,
    'PageRank': pagerank_code,
    'BFS': bfs_code,
}

print("静态特征对比:")
for name, code in queries.items():
    features = extractor.extract(code)
    print(f"{name}: loops={features[0]}, branches={features[2]}, atomic={features[5]}")
```

### 示例 3：从文件加载

```python
# 从文件加载查询代码和图
manager = FeatureManager()

features = manager.extract_from_files(
    query_file='queries/subgraph_iso.py',
    graph_file='data/twitter.gml',
    config=config,
    partition_file='partitions/twitter_partitions.json'
)
```

---

## 特征重要性分析

```python
from autoconfig import CostPredictor

# 训练模型
predictor = CostPredictor()
predictor.train(queries, graphs, configs, times)

# 获取特征重要性
importance = predictor.get_feature_importance()
names = predictor.get_feature_names()

# 排序显示
sorted_idx = importance.argsort()[::-1]
print("Top 10 重要特征:")
for i in sorted_idx[:10]:
    print(f"  {names[i]}: {importance[i]:.4f}")
```

---

## 特征组说明

```python
manager = FeatureManager()
groups = manager.get_feature_groups()

for group_name, feature_names in groups.items():
    print(f"\n{group_name.upper()} ({len(feature_names)} features):")
    for name in feature_names:
        print(f"  - {name}")
```

输出：
```
STATIC (8 features):
  - static_loop_count
  - static_max_loop_depth
  - static_branch_count
  ...

SYMBOLIC (12 features):
  - sym_vscan_coeff
  - sym_vscan_requires
  - sym_escan_coeff
  ...

GRAPH_PARTITION (17 features):
  - graph_num_vertices
  - graph_num_edges
  - graph_diameter
  ...

CONFIG (10 features):
  - conf_memory_limit
  - conf_num_threads
  ...
```

---

## 运行示例

```bash
# 运行完整的特征提取示例
cd autoconfig
source venv/bin/activate
python autoconfig/examples/feature_extraction_example.py
```

---

## 参考资料

- [使用指南](usage_guide.md) - 详细 API 和使用方法
- [API 参考](api_reference.md) - 类和函数文档
- [快速开始](quickstart.md) - 5 分钟入门
