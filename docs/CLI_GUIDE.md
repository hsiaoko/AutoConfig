# AutoConfig 命令行工具使用指南

## 概述

AutoConfig 提供三个独立的特征提取工具和一个完整的数据准备管道：

1. **query** - 从图查询代码提取特征
2. **graph** - 从图数据（边列表）提取特征
3. **config** - 使用 LHS 方法生成配置样本
4. **all** - 完整的数据准备管道

所有输出均为 YAML 格式，保存在 `out/` 目录下。

---

## 安装

```bash
cd autoconfig
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 1. 查询代码特征提取

从图查询源代码提取**静态特征**和**符号特征**。

### 命令

```bash
autoconfig query --input <查询文件> --output <输出 YAML>
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input`, `-i` | 查询源代码文件（必需） | - |
| `--output`, `-o` | 输出 YAML 文件路径 | `out/query_features.yaml` |
| `--num-vertices` | 预估顶点数（用于符号特征） | - |
| `--num-edges` | 预估边数（用于符号特征） | - |
| `--diameter` | 预估图直径 | `5` |

### 示例

```bash
# 基本用法
autoconfig query --input queries/bfs.py --output out/query_features.yaml

# 带图统计信息
autoconfig query \
    --input queries/bfs.py \
    --output out/query_features.yaml \
    --num-vertices 10000 \
    --num-edges 50000 \
    --diameter 6
```

### 输出示例

```yaml
feature_count:
  static: 8
  symbolic: 12
  total: 20
metadata:
  input_file: queries/bfs.py
query_features:
  static:
    static_loop_count: 2.0
    static_max_loop_depth: 1.0
    static_branch_count: 1.0
    static_variable_count: 4.0
    static_recursion_count: 0.0
    static_atomic_op_count: 0.0
    static_sync_count: 0.0
    static_explicit_parallel_flag: 0.0
  symbolic:
    sym_vscan_coeff: 0.0
    sym_vscan_requires: 0.0
    sym_escan_coeff: 50000.0
    sym_escan_requires: 1.0
    sym_fscan_coeff: 50000.0
    sym_fscan_requires: 1.0
    sym_rexp_coeff: 0.0
    sym_rexp_requires: 0.0
    sym_atom_coeff: 0.0
    sym_atom_requires: 0.0
    sym_comm_coeff: 0.0
    sym_comm_requires: 0.0
```

### 支持的语言模式

- **Python**: `for v in G.vertices():`, `while cond:`
- **C/C++**: `for(;;)`, `while()`, `if()`
- **伪代码**: `for v in Vertices:`, `if !condition:`

---

## 2. 图数据特征提取

从边列表格式的图数据提取特征，支持**单图**和**分图**。

### 命令

```bash
autoconfig graph --input <边列表文件或文件夹> --output <输出 YAML>
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input`, `-i` | 边列表 CSV 文件或分图文件夹（必需） | - |
| `--output`, `-o` | 输出 YAML 文件路径 | `out/graph_features.yaml` |

### 输入格式

边列表 CSV 格式：

```csv
src,dst
0,1
0,2
1,2
2,3
```

第一行可以是标题行（`src,dst`）或直接是数据。

### 示例

#### 单图

```bash
autoconfig graph --input data/twitter_edges.csv --output out/graph_features.yaml
```

#### 分图（文件夹）

```bash
autoconfig graph --input data/partitions/ --output out/graph_features.yaml
```

文件夹内应包含多个边列表文件（`*.csv`, `*.edges`, `*.txt`）。

### 输出示例（单图）

```yaml
graph_features:
  basic:
    num_vertices: 1000
    num_edges: 5000
    density: 0.005
  degree:
    avg: 10.0
    max: 50.0
    min: 2.0
    std: 5.5
    skew: 1.8
  structure:
    diameter: 6
    clustering_coeff: 0.15
    num_components: 1
  partition:
    num_partitions: 1
    boundary_vertices: 0
    edge_cut_ratio: 0.0
    balance: 1.0
metadata:
  input_file: data/twitter_edges.csv
  input_type: single_graph
  num_edges_loaded: 5000
```

### 输出示例（分图）

```yaml
graph_features:
  basic:
    num_vertices: 1000
    num_edges: 5000
  partition:
    num_partitions: 4
    boundary_vertices: 150
    boundary_degree_sum: 450.0
    avg_partition_size: 250.0
    partition_size_std: 10.5
    edge_cut_ratio: 0.25
    balance: 0.95
  quality_metrics:
    edge_cut_ratio: 0.25
    balance_score: 0.95
    boundary_ratio: 0.15
    comprehensive_score: 0.22
partition_info:
  '0':
    file: data/partitions/partition_0.csv
    num_edges: 1200
    num_vertices: 245
  '1':
    file: data/partitions/partition_1.csv
    num_edges: 1300
    num_vertices: 255
  ...
metadata:
  input_folder: data/partitions/
  input_type: partitioned_graph
  num_partitions: 4
  total_edges: 5000
```

---

## 3. 配置生成（LHS 采样）

使用**拉丁超立方采样**（Latin Hypercube Sampling）从资源目录生成配置样本。

### 命令

```bash
autoconfig config --num-samples <数量> --output <输出 YAML> [ --resource-catalog <目录文件> | --use-default-catalog ]
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--resource-catalog`, `-c` | 资源目录 YAML 文件 | - |
| `--num-samples`, `-n` | 生成的配置数量 | `20` |
| `--k-min` | 最小实例数 | `1` |
| `--k-max` | 最大实例数 | `16` |
| `--output`, `-o` | 输出 YAML 文件路径 | `out/config_features.yaml` |
| `--use-default-catalog` | 使用默认云资源目录 | `False` |

### 资源目录格式

```yaml
resources:
  - cpu_cores: 4
    memory_gb: 8
    storage_gb: 100
    num_gpus: 0

  - cpu_cores: 16
    memory_gb: 64
    storage_gb: 500
    num_gpus: 1
    gpu_memory_gb: 16

  - cpu_cores: 32
    memory_gb: 128
    storage_gb: 1000
    num_gpus: 4
    gpu_memory_gb: 32
```

### 示例

```bash
# 使用默认目录
autoconfig config \
    --num-samples 20 \
    --output out/configs.yaml \
    --use-default-catalog

# 使用自定义目录
autoconfig config \
    --resource-catalog resources.yaml \
    --num-samples 30 \
    --k-min 2 \
    --k-max 8 \
    --output out/configs.yaml
```

### 输出示例

```yaml
configurations:
  - config_id: 0
    k: 4
    resource:
      cpu_cores: 16
      memory_gb: 64
      storage_gb: 500
      num_gpus: 1
  - config_id: 1
    k: 8
    resource:
      cpu_cores: 32
      memory_gb: 128
      storage_gb: 1000
      num_gpus: 4
  ...
config_features:
  - conf_k_instances: 4
    conf_total_cpu_cores: 64
    conf_total_memory_gb: 256
    conf_total_storage_gb: 2000
    conf_total_gpus: 4
    conf_per_instance:
      cpu_cores: 16
      memory_gb: 64
      ...
metadata:
  num_samples: 20
  k_range: [1, 16]
  catalog_size: 11
  sampling_method: latin_hypercube
catalog_stats:
  cpu_cores:
    min: 2
    max: 64
    mean: 18.5
    std: 15.2
  ...
```

---

## 4. 完整管道

一次性提取查询、图、配置的所有特征。

### 命令

```bash
autoconfig all --query <查询文件> --graph <图文件/文件夹> --config-n <数量> --output <输出目录>
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--query`, `-q` | 查询源代码文件 | - |
| `--graph`, `-g` | 图边列表文件或文件夹 | - |
| `--config-n` | 配置样本数量 | `20` |
| `--k-min` | 最小实例数 | `1` |
| `--k-max` | 最大实例数 | `16` |
| `--output`, `-o` | 输出目录 | `out/` |

### 示例

```bash
autoconfig all \
    --query queries/bfs.py \
    --graph data/twitter_edges.csv \
    --config-n 20 \
    --output out/
```

### 输出

```
out/
├── query_features.yaml    # 查询特征
├── graph_features.yaml    # 图特征
└── config_features.yaml   # 配置特征
```

---

## 使用场景

### 场景 1: 准备训练数据

```bash
# 1. 提取多个查询的特征
for query in queries/*.py; do
    autoconfig query --input $query --output out/queries/$(basename $query .py).yaml
done

# 2. 提取图特征
autoconfig graph --input data/graph.csv --output out/graph.yaml

# 3. 生成配置样本
autoconfig config --num-samples 50 --output out/configs.yaml --use-default-catalog
```

### 场景 2: 分析查询复杂度

```bash
autoconfig query \
    --input queries/pagerank.py \
    --output out/pagerank.yaml \
    --num-vertices 1000000 \
    --num-edges 50000000

# 查看符号特征了解主要计算模式
# - sym_escan_coeff: 边扫描工作量
# - sym_fscan_coeff: 前沿迭代工作量
# - sym_atom_coeff: 原子操作竞争
```

### 场景 3: 分析分区质量

```bash
autoconfig graph \
    --input data/partitions/ \
    --output out/partitions.yaml

# 查看质量指标
# - edge_cut_ratio: 边切割比例（越低越好）
# - balance: 分区平衡度（越接近 1 越好）
# - comprehensive_score: 综合评分（越低越好）
```

### 场景 4: 配置空间探索

```bash
# 生成覆盖整个配置空间的样本
autoconfig config \
    --num-samples 100 \
    --k-min 1 \
    --k-max 32 \
    --resource-catalog cloud_catalog.yaml \
    --output out/large_config_space.yaml
```

---

## 输出文件结构

### 查询特征 YAML

```yaml
feature_count:
  static: 8
  symbolic: 12
  total: 20
metadata:
  input_file: ...
query_features:
  static: { ... }
  symbolic: { ... }
```

### 图特征 YAML

```yaml
graph_features:
  basic: { ... }
  degree: { ... }
  structure: { ... }
  partition: { ... }
  quality_metrics: { ... }  # 仅分图
metadata:
  input_file: ...
  input_type: single_graph|partitioned_graph
partition_info: { ... }  # 仅分图
```

### 配置特征 YAML

```yaml
configurations: [ ... ]
config_features: [ ... ]
metadata:
  num_samples: ...
  k_range: [...]
  catalog_size: ...
  sampling_method: latin_hypercube
catalog_stats: { ... }
```

---

## 常见问题

### Q: 边列表格式不正确怎么办？

确保 CSV 格式为：
```csv
src,dst
0,1
1,2
```

或使用无标题格式：
```csv
0,1
1,2
2,3
```

### Q: 如何自定义资源目录？

创建 YAML 文件：
```yaml
resources:
  - cpu_cores: 8
    memory_gb: 16
    storage_gb: 200
    num_gpus: 0
  ...
```

然后使用 `--resource-catalog my_resources.yaml`。

### Q: LHS 采样的优势是什么？

拉丁超立方采样确保：
1. **空间覆盖**: 每个维度都被均匀采样
2. **样本效率**: 用较少样本覆盖整个空间
3. **避免冗余**: 不会在某个区域过度采样

### Q: 输出 YAML 中的 numpy 类型怎么处理？

使用 `pyyaml` 加载时会自动转换：
```python
import yaml
with open('out/config_features.yaml') as f:
    data = yaml.safe_load(f)
```

---

## 参考资料

- [特征提取方法](docs/feature_extraction.md) - 详细的特征提取原理
- [API 参考](docs/api_reference.md) - Python API 文档
- [快速开始](docs/quickstart.md) - 5 分钟入门
