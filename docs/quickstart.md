# AutoConfig quick start

## Five-minute intro

### 1. Install

```bash
cd AutoConfig
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run the demo driver

```bash
python -m autoconfig.main
```

Example output:

```
============================================================
Graph Query Execution Time Prediction System
============================================================
Generating synthetic data...
Training model...
...
Test Results:
  MAE: ...
  RMSE: ...
  R²: ...
```

### 3. Code sketch

```python
from autoconfig import CostPredictor
import networkx as nx

predictor = CostPredictor()

queries = [...]   # query graphs
graphs = [...]    # data graphs
configs = [...]   # config dicts
times = [...]     # measured runtimes

predictor.train(queries, graphs, configs, times, verbose=True)

query = nx.erdos_renyi_graph(10, 0.2)
graph = nx.erdos_renyi_graph(100, 0.1)
config = {'memory_limit': 8192, 'num_threads': 4, ...}

time = predictor.predict(query, graph, config)
print(f"Predicted time: {time:.2f} ms")
```

## Next steps

- [usage_guide.md](usage_guide.md) — workflow and Python API
- [api_reference.md](api_reference.md) — classes and methods
- [CLI_GUIDE.md](CLI_GUIDE.md) — `autoconfig` CLI
