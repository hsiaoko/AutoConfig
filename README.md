# AutoConfig - 图查询执行时间预测

基于机器学习的图查询执行时间预测系统

## 快速开始

### 安装

```bash
cd autoconfig
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 使用命令行工具

```bash
# 1. 提取查询代码特征（使用占位符）
autoconfig query --input data/queries/gar_match.cu --output out/query_features.yaml

# 2. 提取图数据特征
autoconfig graph --input data/edges.csv --output out/graph_features.yaml

# 3. 生成配置样本（LHS 采样）
autoconfig config --num-samples 20 --output out/config_features.yaml --use-default-catalog

# 4. 合并所有特征（实例化符号特征）
autoconfig merge \
    --query out/query_features.yaml \
    --graph out/graph_features.yaml \
    --config out/config_features.yaml \
    --output out/merged_features.yaml
```

**完整管道**:
```bash
# 一键完成所有步骤
autoconfig all --query query.cu --graph data/ --config-n 20 --output out/
```

---

## 功能特性

### 1. 查询代码特征提取

从图查询源代码提取 20 个特征（8 个静态 + 12 个符号）：

```bash
autoconfig query --input bfs.py --output out/query.yaml
```

**静态特征**: 循环、分支、递归、原子操作、同步等

**符号特征**: VScan、EScan、FScan、RExp、Atom、Comm（用图统计实例化）

### 2. 图数据特征提取

从边列表格式提取图特征，支持单图和分图：

```bash
# 单图
autoconfig graph --input graph.csv --output out/graph.yaml

# 分图（文件夹）
autoconfig graph --input partitions/ --output out/graph.yaml
```

**输出**: 顶点数、边数、度统计、直径、聚类系数、分区质量指标等

### 3. 配置生成（LHS 采样）

使用拉丁超立方采样生成配置样本：

```bash
autoconfig config --num-samples 20 --output out/configs.yaml --use-default-catalog
```

**配置**: (k 实例数，资源类型) 元组，覆盖 CPU、内存、存储、GPU 维度

---

## 文档

| Document | Description |
|----------|-------------|
| [中文使用指南](docs/USAGE_GUIDE_CN.md) | **完整的中文使用文档** |
| [CLI Guide](docs/CLI_GUIDE.md) | Command-line tools detailed guide |
| [Feature Extraction](docs/FEATURE_EXTRACTION.md) | Feature extraction methods (English) |
| [Usage Guide](docs/usage_guide.md) | Python API usage |
| [API Reference](docs/api_reference.md) | Class and function documentation |
| [Quick Start](docs/quickstart.md) | 5-minute introduction |

---

## 项目结构

```
autoconfig/
├── autoconfig/              # 主包
│   ├── __init__.py
│   ├── main.py              # 训练管道
│   ├── cli.py               # 命令行接口
│   ├── feature_extractor/   # 特征提取模块
│   │   ├── static_extractor.py
│   │   ├── symbolic_extractor.py
│   │   ├── graph_partition_extractor.py
│   │   └── feature_manager.py
│   ├── models/              # 模型模块
│   │   └── bayesian_model.py
│   ├── prediction/          # 预测模块
│   │   └── cost_predictor.py
│   └── utils/               # 工具模块
│       ├── query_feature_extractor.py
│       ├── graph_feature_extractor.py
│       └── config_generator.py
├── examples/                # 示例数据
│   ├── query_bfs.py
│   ├── graph_small.csv
│   └── partitions/
├── out/                     # 输出目录
├── docs/                    # 文档
├── resources_template.yaml  # 资源目录模板
├── requirements.txt
└── README.md
```

---

## 示例

### 查询代码特征

```python
# examples/query_bfs.py
def BFS(Graph G, vertex source):
    worklist = [source]
    visited[source] = true
    
    while !worklist.empty():
        for v in worklist:
            for neighbor in G.neighbors(v):
                if !visited[neighbor]:
                    visited[neighbor] = true
                    worklist.append(neighbor)
```

运行：
```bash
autoconfig query --input examples/query_bfs.py --output out/query.yaml
```

输出：
```yaml
query_features:
  static:
    static_loop_count: 2.0
    static_branch_count: 1.0
    ...
  symbolic:
    sym_escan_coeff: 5000.0  # 边扫描
    sym_fscan_coeff: 5000.0  # 前沿迭代
    ...
```

### 图数据特征

边列表格式：
```csv
src,dst
0,1
0,2
1,2
2,3
```

运行：
```bash
autoconfig graph --input examples/graph_small.csv --output out/graph.yaml
```

输出：
```yaml
graph_features:
  basic:
    num_vertices: 10
    num_edges: 17
  degree:
    avg: 3.4
    max: 4.0
    skew: 1.18
  structure:
    diameter: 5
    clustering_coeff: 0.63
  ...
```

### 配置生成

运行：
```bash
autoconfig config --num-samples 10 --output out/configs.yaml --use-default-catalog
```

输出：
```yaml
configurations:
  - config_id: 0
    k: 4
    resource:
      cpu_cores: 16
      memory_gb: 64
      num_gpus: 1
  - config_id: 1
    k: 8
    resource:
      cpu_cores: 32
      memory_gb: 128
      num_gpus: 4
  ...
```

---

## 训练预测模型

```bash
# 使用提取的特征训练贝叶斯模型
autoconfig-train --n-train 100 --n-test 20
```

或使用 Python API：

```python
from autoconfig import CostPredictor

predictor = CostPredictor()
predictor.train(queries, graphs, configs, times)
```

---

## 参考资料

- 特征提取方法基于 Hybrid 图查询成本估计研究
- 贝叶斯模型参考数据库调优工作（Bayesian Optimization, DBTune 等）
- LHS 采样方法用于高效配置空间探索

---

## 许可证

MIT License
