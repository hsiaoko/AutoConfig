# AutoConfig — 图查询执行时间预测

基于特征抽取与机器学习的图查询执行时间 / 成本估计，支持论文中的 **静态特征**、**符号模板（公式）**、**图与划分统计**、**系统配置** 四元组流水线。

---

## 安装

在仓库根目录（含 `pyproject.toml` / `requirements.txt`）执行：

```bash
cd AutoConfig
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

安装完成后可使用命令行入口 `autoconfig`，或直接用 `python experiments/scripts/step*.py` 跑四步流水线。

---

## 特征流水线（四步 YAML）

与论文 §5 一致：**查询 YAML**（静态为数值、符号为模板与公式）→ **图 YAML** → **系统配置 YAML** → **合并为最终数值特征矩阵**。

| 步骤 | 作用 | 输出文件 |
|------|------|----------|
| 1 | 从查询源码抽取 `Φ_static` 与符号模板 `Φ_sym`（`format_version: 2`，含公式，不含图上的最终系数） | `query_features.yaml` |
| 2 | 从边列表（或分片目录）计算图与划分统计 | `graph_features.yaml` |
| 3 | LHS 等资源采样生成候选配置 | `config_features.yaml` |
| 4 | 用图统计**实例化**符号模板，并与静态 / 图 / 配置拼成向量 | `merged_features.yaml` |

### 方式 A：四段独立脚本（推荐对照论文）

在仓库根目录执行：

```bash
python experiments/scripts/step1_query_features.py \
  -i data/queries/your_query.py -o out/query_features.yaml

python experiments/scripts/step2_graph_features.py \
  -i data/edges.csv -o out/graph_features.yaml

python experiments/scripts/step3_system_config.py \
  -n 20 -o out/config_features.yaml --use-default-catalog

python experiments/scripts/step4_merge_features.py \
  -q out/query_features.yaml \
  -g out/graph_features.yaml \
  -c out/config_features.yaml \
  -o out/merged_features.yaml
```

### 方式 B：`autoconfig` 子命令

```bash
autoconfig query  --input data/queries/gar_match.cu --output out/query_features.yaml
autoconfig graph  --input data/edges.csv --output out/graph_features.yaml
autoconfig config --num-samples 20 --output out/config_features.yaml --use-default-catalog
autoconfig merge   --query out/query_features.yaml --graph out/graph_features.yaml \
                   --config out/config_features.yaml --output out/merged_features.yaml
```

一键生成前三类产物（不含第四步合并）：

```bash
autoconfig all --query query.cu --graph data/ --config-n 20 --output out/
```

---

## 符号特征说明

- **Step 1 的 YAML** 中，`symbolic` 为 **模板层**：每个模式（VScan、EScan、FScan、RExp、Atom、Comm）包含 `template`、`formula`、`detected` 以及所需图/划分字段说明，**不是**最终 `|V|`、`|E|` 等标量。
- **Step 4** 读取 `graph_features.yaml`，由 `FeatureMerger` 将模板实例化为 **12 维** 数值特征（`sym_*_coeff` / `sym_*_requires`），供下游 MLP 使用。
- 若仍使用旧版「扁平 `sym_*` 占位」查询 YAML，合并器仍兼容。

详细公式与字段表见 **[docs/feature_extraction.md](docs/feature_extraction.md)**，四步说明见 **[docs/FEATURE_PIPELINE.md](docs/FEATURE_PIPELINE.md)**。

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [docs/README.md](docs/README.md) | 文档导航 |
| [docs/FEATURE_PIPELINE.md](docs/FEATURE_PIPELINE.md) | 四步流水线（英文） |
| [docs/feature_extraction.md](docs/feature_extraction.md) | 特征定义与合并后维度 |
| [docs/USAGE_GUIDE_CN.md](docs/USAGE_GUIDE_CN.md) | 中文使用指南 |
| [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) | CLI 详解 |
| [docs/api_reference.md](docs/api_reference.md) | API 参考 |

---

## 功能概览

### 查询特征

- **静态**：循环深度、分支、变量、递归、原子操作、同步、显式并行等。
- **符号**：六种 workload 模板 + 公式字符串；在 merge 阶段用图统计求值。

### 图特征

支持单边列表 CSV 或 **分片目录**（多文件），输出顶点/边数、度分布、直径、聚类系数、划分与边界等（见 `graph_features.yaml` 结构）。

### 配置生成

对资源目录做 Latin Hypercube 采样，得到 `(k, resource)` 候选集合，写入 `config_features.yaml`。

---

## 项目结构（节选）

```
AutoConfig/
├── autoconfig/                 # 主包
│   ├── cli.py
│   ├── feature_extractor/      # static / symbolic / graph_partition
│   └── utils/
│       ├── query_feature_extractor.py
│       ├── graph_feature_extractor.py
│       ├── config_generator.py
│       └── feature_merger.py
├── experiments/
│   └── scripts/
│       ├── step1_query_features.py
│       ├── step2_graph_features.py
│       ├── step3_system_config.py
│       └── step4_merge_features.py
├── docs/                       # 说明文档
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 训练与推荐（简述）

```bash
autoconfig train --n-samples 500 --output data/models/
```

```bash
autoconfig recommend --query my_algorithm.cu --graph data/graph.csv --top-n 3 --output out/recommendation.yaml
```

更多参数与实验脚本见 [experiments/README.md](experiments/README.md)。

---

## 多 Agent 开发系统（可选）

参见 [dev_harness/README.md](dev_harness/README.md)、[dev_harness/QUICKSTART.md](dev_harness/QUICKSTART.md)。

---

## 许可证

MIT License
