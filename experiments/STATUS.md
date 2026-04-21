# 实验框架状态总结

## ✅ 已完成

### 1. 完整的实验框架结构
```
experiments/
├── config.yaml              ✓ 主配置文件
├── README.md                ✓ 详细文档
├── run_experiments.sh       ✓ Shell启动脚本
├── verify_config.py         ✓ 验证脚本
├── __init__.py
├── baselines/
│   ├── __init__.py
│   └── baselines.py         ✓ BO, RL, GPTuner, BestConfig, Oracle
├── scripts/
│   ├── run_all_experiments.py      ✓ 主运行脚本
│   ├── run_single_experiment.py    ✓ 单个实验运行
│   ├── exp1_feature_extraction.py  ✓ Exp-1: 特征提取有效性
│   ├── exp2_effectiveness.py       ✓ Exp-2: 效果性
│   ├── exp3_robustness.py         ✓ Exp-3: 鲁棒性
│   ├── exp5_scalability.py        ✓ Exp-5: 可扩展性和消融
│   ├── exp6_case_study.py         ✓ Exp-6: 案例研究
│   └── visualize_results.py       ✓ 结果可视化
├── workloads/
│   ├── __init__.py
│   └── workloads.py         ✓ WCC, SSSP, PR, BFS, SubIso, GARs
└── results/                 ✓ 输出目录
```

### 2. 实验覆盖的六个维度
- ✅ **Exp-1**: 特征提取有效性（MAE, MAPE, R²）
- ✅ **Exp-2**: 效果性（端到端运行时对比）
- ✅ **Exp-3**: 鲁棒性（预测错误、分布偏移）
- ✅ **Exp-4**: 效率（选择延迟、调优开销）
- ✅ **Exp-5**: 可扩展性和消融（图大小、集群大小、组件影响）
- ✅ **Exp-6**: 案例研究（GARs欺诈检测）

### 3. 支持的数据集
- ✅ IMDB (19M 节点, 0.1B 边)
- ✅ Github (37K 节点, 289K 边)
- ✅ Gowalla (196K 节点, 1.0M 边)
- ✅ DBLP (317K 节点, 1.0M 边)
- ✅ Youtube (1.1M 节点, 2.9M 边)
- ✅ Wikittalk (2.4M 节点, 5.0M 边)
- ✅ Twitter (41.6M 节点, 1.5B 边)
- ✅ Syn-Small/Medium/Large (R-MAT 合成图)

### 4. 工作负载
- ✅ WCC - 弱连通分量
- ✅ SSSP - 单源最短路径
- ✅ PR - PageRank
- ✅ BFS - 广度优先搜索
- ✅ SubIso - 子图同构
- ✅ GARs - 图关联规则（欺诈检测）

### 5. Baseline方法
- ✅ BO - 贝叶斯优化
- ✅ RL - 强化学习
- ✅ GPTuner - LLM增强的BO
- ✅ BestConfig - 网格搜索
- ✅ Oracle - 穷举搜索（上限基准）

### 6. 配置空间
- ✅ k: 1-64（实例数）
- ✅ CPU: 1-256 核心
- ✅ 内存: 16-1024 GB
- ✅ GPU: 0-8个 A100

---

## 📝 已修复的问题

### 1. YAML配置文件语法错误
**问题**: `evaluation_split` 错误缩进在 `datasets` 列表内
**状态**: ✅ 已修复
**修复**: 移动为独立的顶层键

### 2. Python导入问题
**问题**: 相对导入错误
**状态**: ✅ 已修复
**修复**: 使用 `importlib` 动态导入

### 3. 语法错误
**问题**: `exp1_feature_extraction.py` 中有反引号语法错误
**状态**: ✅ 已修复
**修复**: 移除了多余的反引号

### 4. 模块导入错误
**问题**: 实验脚本导入不存在的模块或不正确的相对导入
**状态**: ✅ 已修复
**修复**:
- 移除不存在的 `BayesRidgeModel` 导入
- 修复所有 `from experiments.*` 相对导入
- 使用动态导入跨脚本共享工具函数
- 详情见 `IMPORT_FIXES.md`

---

## 🚀 如何运行实验

### 方式1: 使用Shell脚本（推荐）

```bash
cd /Users/hsiaoko/Projects/AutoConfig/experiments
chmod +x run_experiments.sh
./run_experiments.sh                # 运行所有实验
./run_experiments.sh --exp 1 2 3    # 运行指定实验
```

### 方式2: 使用Python脚本

```bash
cd /Users/hsiaoko/Projects/AutoConfig/experiments/scripts

# 运行所有实验
python run_all_experiments.py --all

# 运行指定实验
python run_all_experiments.py --exp 1 2 3

# 运行单个实验
python run_single_experiment.py --exp 1
```

### 方式3: 自定义配置和输出

```bash
python run_all_experiments.py --all \
  --config my_config.yaml \
  --output-dir my_results/
```

---

## 📊 结果输出

### JSON结果文件
```
experiments/results/
├── exp1_inv_dist_WCC.json
├── exp1_inv_dist_SSSP.json
├── exp1_out_dist_within_class.json
├── exp2ffectiveness.json
├── exp3_error_robustness.json
├── exp3_distribution_shift.json
├── exp3_hamming.json
├── exp4_selection_latency.json
├── exp4_tuning_overhead.json
├── exp5_graph_scaling.json
├── exp5_cluster_scaling.json
├── exp5_ablation.json
├── exp6_configuration_landscape.json
├── exp6_gars_discovery.json
└── exp6_pipeline_stages.json
```

### 可视化图表
```bash
python experiments/scripts/visualize_results.py

# 输出:
# - fig_exp1_feature_extraction.png
# - fig_exp2_effectiveness.png
# - fig_exp3_exp4_robustness_efficiency.png
# - fig_exp5_exp6_scalability_case_study.png
```

---

## 🔧 环境检查

### 验证框架设置

```bash
cd /Users/hsiaoko/Projects/AutoConfig
python experiments/verify_config.py
```

### 检查依赖包

依赖包应该已经安装（需要Anaconda环境）：

```bash
/opt/anaconda3/bin/python -c "import yaml, numpy, scipy, sk learn, pandas, networkx, matplotlib; print('✓ All dependencies available')"
```

---

## ⚠️ 注意事项

### 关于运行时错误

如果遇到段错误（退出码139），可能是由于：

1. **AutoConfig模块导入问题**
   - 解决：实验脚本已修改使用动态导入

2. **依赖包版本不兼容**
   - 解决：使用Anaconda 3.12环境（已验证可用）

3. **内存不足**
   - 解决：在 `config.yaml` 中减少图大小
   ```yaml
   datasets:
     - name: "TestGraph"
       num_vertices: 5000  # 减小规模
   ```

### 推荐运行环境

- **Python**: 3.12（Anaconda）
- **OS**: macOS (Darwin 25.3.0)
- **包管理**: 使用已安装的Anaconda环境

---

## 📖 修改配置

### 自定义数据集大小

编辑 `experiments/config.yaml`:

```yaml
datasets:
  - name: "MyDataset"
    type: "synthetic_rmat"
    num_vertices: 50000  # 自定义大小
    num_edges: 400000
```

### 自定义成本函数权重

```yaml
cost_function:
  w1: 0.7  # 更重视运行时
  w2: 0.3  # 减少货币成本的权重
```

### 调整实验输出目录

```bash
python run_all_experiments.py --all \
  --output-dir /path/to/custom/results/
```

---

## 🤝 贡献和扩展

### 添加新工作负载

1. 编辑 `experiments/workloads/workloads.py`
2. 添加新的查询代码模板
3. 在 `experiments/workloads/__init__.py` 中导出

### 添加新Baseline

1. 在 `experiments/baselines/baselines.py` 中实现新类
2. 继承 `BaselineMethod` 基类
3. 实现 `recommend()` 方法

### 自定义评估指标

在对应的实验脚本中添加新的评估函数。

---

## 📚 参考文档

- [实验框架详细文档](README.md)
- [主项目README](../README.md)
- [论文实验设置](../docs/paper_experiments.md)（如需要创建）

---

## ✅ 下一步

1. **验证框架**: 运行 `verify_config.py` 确认正常
2. **运行单一实验**: 先运行Exp-1测试特定功能
3. **生成可视化**: 运行 `visualize_results.py` 生成图表
4. **完整实验**: 运行所有实验获取完整结果
5. **结果分析**: 比较不同方法和配置的性能

---

## 💡 使用提示

1. **首次运行**建议从单个实验开始了解流程
2. **调整配置**前建议先备份原始 `config.yaml`
3. **大规模实验**前建议先在小规模数据上测试
4. **结果分析**可以使用Jupyter Notebooks进行交互式探索
5. **可视化**图表可直接用于论文或报告

---

**状态**: ✅ 实验框架已完整设置，ready to run!
**最后更新**: 2026-04-03