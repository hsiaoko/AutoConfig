# AutoConfig 使用指南（更新版）

## 概述

AutoConfig 现在支持完整的特征提取和合并管道：

1. **Query 特征提取** - 从查询代码提取静态和符号特征（使用占位符）
2. **Graph 特征提取** - 从图数据提取结构和分区特征
3. **Config 特征生成** - 使用 LHS 生成配置样本
4. **Feature Merge** - 合并所有特征，实例化符号特征

---

## 工作流程

```
1. query.py ─────┐
                 │
2. edges.csv ────┼──→ merge ──→ merged_features.yaml
                 │              (47 维特征向量)
3. configs.yaml ─┘
```

---

## 命令行使用

### 1. 提取查询特征

```bash
autoconfig query --input <查询文件> --output <输出 YAML>
```

**示例**:
```bash
autoconfig query \
    --input queries/gar_match.cu \
    --output out/query_features.yaml
```

**输出**:
```yaml
query_features:
  static:
    static_loop_count: 8.0
    static_atomic_op_count: 4.0
    ...
  symbolic:
    sym_vscan_coeff: 1.0  # 占位符
    sym_vscan_requires: 1.0
    sym_escan_coeff: 1.0  # 占位符
    ...
placeholders:
  note: Graph-dependent values use placeholders...
  required_graph_stats:
    - num_vertices
    - num_edges
    - diameter
    ...
```

### 2. 提取图特征

```bash
# 单图
autoconfig graph --input data/edges.csv --output out/graph_features.yaml

# 分图（文件夹）
autoconfig graph --input data/partitions/ --output out/graph_features.yaml
```

**示例**:
```bash
autoconfig graph \
    --input data/twitter_edges.csv \
    --output out/graph_features.yaml
```

**输出**:
```yaml
graph_features:
  basic:
    num_vertices: 10000
    num_edges: 50000
  degree:
    avg: 10.0
    max: 150.0
    skew: 15.0
  structure:
    diameter: 6
    ...
  partition:
    num_partitions: 4
    edge_cut_ratio: 0.25
    balance: 0.95
```

### 3. 生成配置样本

```bash
autoconfig config \
    --num-samples 20 \
    --output out/config_features.yaml \
    --use-default-catalog
```

**示例**:
```bash
autoconfig config \
    --num-samples 30 \
    --k-min 2 \
    --k-max 8 \
    --resource-catalog resources.yaml \
    --output out/config_features.yaml
```

### 4. 合并特征

```bash
autoconfig merge \
    --query out/query_features.yaml \
    --graph out/graph_features.yaml \
    --config out/config_features.yaml \
    --output out/merged_features.yaml
```

**输出**:
```yaml
feature_matrix:
  - [8.0, 2.0, ..., 0.0]  # 样本 1: 47 维特征
  - [8.0, 2.0, ..., 0.0]  # 样本 2
  ...
feature_names:
  - static_loop_count
  - static_max_loop_depth
  - ...
  - conf_compression_enabled
metadata:
  num_samples: 20
  num_features: 47
```

---

## 完整管道示例

### 示例 1: GAR Match 查询

```bash
cd autoconfig
source venv/bin/activate

# 1. 提取查询特征
autoconfig query \
    --input examples/query_gar_match.cu \
    --output out/query.yaml

# 2. 提取图特征
autoconfig graph \
    --input examples/graph_small.csv \
    --output out/graph.yaml

# 3. 生成配置（3 个样本）
autoconfig config \
    --num-samples 3 \
    --output out/config.yaml \
    --use-default-catalog

# 4. 合并特征
autoconfig merge \
    --query out/query.yaml \
    --graph out/graph.yaml \
    --config out/config.yaml \
    --output out/merged.yaml
```

### 示例 2: 批量处理

```bash
#!/bin/bash
# batch_process.sh

QUERIES=("bfs.cu" "pagerank.cu" "cc.cu")
GRAPHS=("graph1" "graph2")
OUTPUT_DIR="results"

for query in "${QUERIES[@]}"; do
    for graph in "${GRAPHS[@]}"; do
        echo "Processing $query + $graph"
        
        # 提取特征
        autoconfig query --input queries/$query -o out/q_${query%.cu}.yaml
        autoconfig graph --input data/$graph/ -o out/g_${graph}.yaml
        autoconfig config -n 10 -o out/c.yaml --use-default-catalog
        
        # 合并
        autoconfig merge \
            -q out/q_${query%.cu}.yaml \
            -g out/g_${graph}.yaml \
            -c out/c.yaml \
            -o $OUTPUT_DIR/${query%.cu}_${graph}.yaml
    done
done
```

---

## 特征说明

### 特征维度（47 维）

| 组别 | 特征数 | 说明 |
|------|--------|------|
| Static | 8 | 查询代码结构特征 |
| Symbolic | 12 | 符号工作负载模板（实例化后） |
| Graph | 17 | 图和分区统计 |
| Config | 10 | 系统配置参数 |
| **Total** | **47** | |

### 符号特征实例化

合并阶段自动将占位符替换为实际图统计：

| 符号模板 | 占位符 | 实例化公式 |
|---------|--------|-----------|
| VScan | 1.0 | \|V\| |
| EScan | 1.0 | \|E\| |
| FScan | 1.0 | \|E\| (简化：D × \|E\|/D) |
| RExp | 1.0 | avg_degree ^ diameter |
| Atom | 1.0 | \|E\| × skew |
| Comm | 1.0 | boundary_degree_sum |

---

## 输出文件结构

```
out/
├── query_features.yaml      # 查询特征（含占位符）
├── graph_features.yaml      # 图特征
├── config_features.yaml     # 配置特征（多个样本）
└── merged_features.yaml     # 合并后的特征矩阵
```

### merged_features.yaml 结构

```yaml
feature_groups:
  static: 8
  symbolic: 12
  graph: 17
  config: 10

feature_matrix:
  - [8.0, 2.0, 16.0, ..., 0.0]  # 样本 1
  - [8.0, 2.0, 16.0, ..., 0.0]  # 样本 2
  ...

feature_names:
  - static_loop_count
  - static_max_loop_depth
  - ...
  - conf_compression_enabled

metadata:
  num_samples: 20          # 配置样本数
  num_features: 47         # 总特征数
  query_file: ...
  graph_file: ...
  config_file: ...
```

---

## Python API 使用

```python
from autoconfig.utils import (
    QueryFeatureExtractor,
    GraphFeatureExtractor,
    ConfigGenerator,
    FeatureMerger
)

# 1. 提取查询特征
query_ext = QueryFeatureExtractor()
query_features = query_ext.extract_from_file('queries/gar_match.cu')

# 2. 提取图特征
graph_ext = GraphFeatureExtractor()
graph_features = graph_ext.extract_single('data/edges.csv')

# 3. 生成配置
config_gen = ConfigGenerator(catalog)
configs = config_gen.generate(num_samples=20, k_range=(1, 16))

# 4. 合并特征
merger = FeatureMerger()
merged = merger.merge_all(
    'out/query.yaml',
    'out/graph.yaml',
    'out/config.yaml'
)

# 获取特征矩阵
X = np.array(merged['feature_matrix'])  # (20, 47)
```

---

## 常见问题

### Q: 为什么不直接在 query 阶段实例化符号特征？

**A**: 因为查询可能用于不同的图，图特征在另一个阶段生成。使用占位符可以：
1. 解耦查询分析和图分析
2. 支持同一个查询用于多个图
3. 在合并阶段灵活实例化

### Q: 如何自定义资源配置？

**A**: 创建 resources.yaml 文件：
```yaml
resources:
  - cpu_cores: 8
    memory_gb: 32
    storage_gb: 200
    num_gpus: 1
  ...
```

然后使用 `--resource-catalog resources.yaml`。

### Q: 如何修改特征顺序？

**A**: 修改 `FeatureMerger.__init__` 中的 `feature_order` 列表。

### Q: 如何添加新的特征类型？

**A**: 
1. 在对应的 extractor 中添加特征提取逻辑
2. 在 `FeatureMerger.feature_order` 中添加特征名
3. 更新 `feature_groups` 计数

---

## 下一步

1. **训练模型**: 使用 merged_features.yaml 中的特征矩阵训练贝叶斯模型
2. **特征选择**: 分析特征重要性，选择关键特征
3. **超参数调优**: 优化模型性能

```python
from autoconfig import CostPredictor

predictor = CostPredictor()
predictor.train(queries, graphs, configs, times)
```

---

## 参考资料

- [CLI Guide](docs/CLI_GUIDE.md) - 命令行详细文档
- [Feature Extraction](docs/FEATURE_EXTRACTION.md) - 特征提取方法（英文）
- [API Reference](docs/api_reference.md) - API 文档
