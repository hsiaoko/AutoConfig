# AutoConfig 快速开始

## 5 分钟入门

### 1. 安装

```bash
cd autoconfig
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. 运行示例

```bash
python autoconfig/main.py
```

输出：
```
============================================================
Graph Query Execution Time Prediction System
============================================================
Generating synthetic data...

Training model...
...
Test Results:
  MAE: xxx
  RMSE: xxx
  R²: xxx

Prediction Example:
  Predicted Execution Time: xxx ms
```

### 3. 代码示例

```python
from autoconfig import CostPredictor
import networkx as nx

# 创建预测器
predictor = CostPredictor()

# 准备你的数据
queries = [...]  # 查询图列表
graphs = [...]   # 数据图列表  
configs = [...]  # 配置列表
times = [...]    # 实际执行时间

# 训练
predictor.train(queries, graphs, configs, times, verbose=True)

# 预测
query = nx.erdos_renyi_graph(10, 0.2)
graph = nx.erdos_renyi_graph(100, 0.1)
config = {'memory_limit': 8192, 'num_threads': 4, ...}

time = predictor.predict(query, graph, config)
print(f"预计时间：{time:.2f} ms")
```

## 下一步

- [详细使用指南](usage_guide.md) - 完整 API 和示例
- [API 参考](api_reference.md) - 类和方法文档
