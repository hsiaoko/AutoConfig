# AutoConfig 快速开始指南

## 5 分钟快速上手

### 安装

```bash
cd autoconfig
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 完整流程示例

```bash
# 一键运行完整流程
bash examples/run_full_pipeline.sh
```

这会执行：
1. ✅ 提取查询代码特征 (C++/CUDA)
2. ✅ 提取图数据特征
3. ✅ 生成容量受限的配置
4. ✅ 合并所有特征为 47 维特征向量

### 分步使用

#### 1. 提取查询特征

```bash
autoconfig query --input examples/query_gar_match.cu -o out/query.yaml
```

**输出**: 20 个特征 (8 静态 + 12 符号占位符)

#### 2. 提取图特征

```bash
autoconfig graph --input data/graph_medium_pl.csv -o out/graph.yaml
```

**输出**: 17 个图特征 (顶点、边、度分布、聚类等)

#### 3. 生成配置

```bash
# 使用系统容量文件
autoconfig config \
    --capacity data/conf/system_capacity_small.yaml \
    -n 5 \
    -o out/config.yaml
```

**输出**: 5 个配置样本，每个包含 k 台机器的资源

#### 4. 合并特征

```bash
autoconfig merge \
    --query out/query.yaml \
    --graph out/graph.yaml \
    --config out/config.yaml \
    -o out/merged.yaml
```

**输出**: 47 维特征矩阵 (5 样本 × 47 特征)

---

## 数据文件

### 查询代码示例

- `examples/query_gar_match.cu` - GAR Match CUDA 内核代码

### 图数据示例

| 文件 | 节点 | 边 | 类型 |
|------|------|-----|------|
| `data/graph_small_*.csv` | 100 | ~250 | 小型图 |
| `data/graph_medium_*.csv` | 1,000 | ~5,000 | 中型图 |
| `data/graph_large_*.csv` | 5,000 | ~50,000 | 大型图 |

### 系统容量配置

| 文件 | CPU | 内存 | GPU | 最大机器数 |
|------|-----|------|-----|-----------|
| `data/conf/system_capacity_small.yaml` | 64C | 256GB | 4 | 4 |
| `data/conf/system_capacity_medium.yaml` | 256C | 1TB | 16 | 8 |
| `data/conf/system_capacity_large.yaml` | 512C | 2TB | 32 | 8 |
| `data/conf/system_capacity_cpu_only.yaml` | 128C | 512GB | 0 | 4 |

---

## 命令行工具

### query - 提取查询特征

```bash
autoconfig query --input <代码文件> -o <输出 YAML>
```

**参数**:
- `--input`, `-i`: 查询源代码文件 (必需)
- `--output`, `-o`: 输出 YAML 文件 (默认：out/query_features.yaml)

### graph - 提取图特征

```bash
autoconfig graph --input <边列表 CSV 或文件夹> -o <输出 YAML>
```

**参数**:
- `--input`, `-i`: 边列表文件或分图文件夹 (必需)
- `--output`, `-o`: 输出 YAML 文件 (默认：out/graph_features.yaml)

### config - 生成配置

```bash
# 方式 1: 使用系统容量 (推荐)
autoconfig config --capacity <容量 YAML> -n <样本数> -o <输出>

# 方式 2: 使用默认目录
autoconfig config -n 20 --use-default-catalog -o <输出>

# 方式 3: 命令行指定资源
autoconfig config --cpu 16 --memory 64 -n 10 -o <输出>
```

**参数**:
- `--capacity`: 系统容量 YAML 文件
- `--num-samples`, `-n`: 生成的配置样本数 (默认：20)
- `--k-min`: 最小机器数 (默认：1)
- `--k-max`: 最大机器数 (默认：容量限制)

### merge - 合并特征

```bash
autoconfig merge \
    --query <查询 YAML> \
    --graph <图 YAML> \
    --config <配置 YAML> \
    -o <输出>
```

**参数**:
- `--query`, `-q`: 查询特征 YAML (必需)
- `--graph`, `-g`: 图特征 YAML (必需)
- `--config`, `-c`: 配置特征 YAML (必需)
- `--output`, `-o`: 输出合并后的 YAML (默认：out/merged_features.yaml)

---

## 输出示例

### 查询特征 (query_features.yaml)

```yaml
query_features:
  static:
    static_loop_count: 8.0
    static_atomic_op_count: 4.0
    static_sync_count: 7.0
  symbolic:
    sym_vscan_coeff: 1.0  # 占位符
    sym_escan_coeff: 1.0  # 占位符
    sym_atom_coeff: 1.0   # 占位符
```

### 图特征 (graph_features.yaml)

```yaml
graph_features:
  basic:
    num_vertices: 1000
    num_edges: 4975
  degree:
    avg: 9.95
    max: 144.0
    skew: 14.47
  structure:
    diameter: 5
    clustering_coeff: 0.042
```

### 配置特征 (config_features.yaml)

```yaml
configurations:
  - config_id: 0
    k: 1
    resource:
      cpu_cores: 16
      memory_gb: 64
      num_gpus: 1
  
  - config_id: 1
    k: 3
    resource:
      cpu_cores: 48
      memory_gb: 192
      num_gpus: 3

config_features:
  - conf_k_instances: 1
    conf_total_cpu_cores: 16
    conf_total_memory_gb: 64
    ...
```

### 合并特征 (merged_features.yaml)

```yaml
feature_matrix:
  - [8.0, 2.0, ..., 64.0, ...]  # 样本 1: 47 维
  - [8.0, 2.0, ..., 192.0, ...] # 样本 2
  ...

feature_names:
  - static_loop_count
  - static_max_loop_depth
  - ...
  - conf_compression_enabled

metadata:
  num_samples: 5
  num_features: 47
```

---

## 特征维度

| 组别 | 特征数 | 说明 |
|------|--------|------|
| Static | 8 | 查询代码结构 |
| Symbolic | 12 | 符号工作负载 (实例化后) |
| Graph | 17 | 图和分区统计 |
| Config | 10 | 系统配置参数 |
| **Total** | **47** | |

---

## 下一步

### 训练模型

使用生成的特征矩阵训练贝叶斯模型：

```python
from autoconfig import CostPredictor
import numpy as np
import yaml

# 加载特征
with open('out/merged_features.yaml') as f:
    data = yaml.safe_load(f)

X = np.array(data['feature_matrix'])  # (5, 47)
y = np.array([...])  # 实际执行时间 (需要测量)

# 训练
predictor = CostPredictor()
predictor.train(queries, graphs, configs, y)
```

### 预测执行时间

```python
# 预测新查询的执行时间
pred = predictor.predict(new_query, new_graph, new_config)
print(f"Predicted time: {pred:.2f} ms")
```

---

## 常见问题

### Q: 如何生成更多配置样本？

**A**: 增加 `-n` 参数：

```bash
autoconfig config --capacity system.yaml -n 50 -o out/configs.yaml
```

### Q: 如何自定义系统容量？

**A**: 创建自定义 YAML 文件：

```yaml
total_cpu_cores: 128
total_memory_gb: 512
total_gpus: 8
machine:
  cpu_cores: 32
  memory_gb: 128
  num_gpus: 2
```

### Q: 特征中的占位符是什么？

**A**: 查询特征中的符号特征使用占位符 (1.0)，在 merge 阶段会用实际图统计实例化。

### Q: 如何查看生成的配置？

**A**: 查看输出 YAML 文件：

```bash
cat out/config_features.yaml | head -50
```

---

## 参考资料

- [CLI 使用指南](docs/CLI_GUIDE.md) - 命令行详细文档
- [配置生成指南](docs/CONFIG_GUIDE_CN.md) - 配置生成详解
- [特征提取方法](docs/FEATURE_EXTRACTION.md) - 特征提取原理
- [中文使用指南](docs/USAGE_GUIDE_CN.md) - 完整中文文档
