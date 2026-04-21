# AutoConfig architecture

## Overview

AutoConfig is a machine-learning system for **graph query configuration**: it analyzes query patterns, graph statistics, and resource settings to predict runtime and cost and to recommend configurations.

## System diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        AutoConfig                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │   Offline training   │         │   Online serving    │       │
│  │  ┌───────────────┐   │         │  ┌───────────────┐   │       │
│  │  │ DataGenerator │   │         │  │ CostPredictor│   │       │
│  │  └───────┬───────┘   │         │  └───────┬───────┘   │       │
│  │  ┌───────▼───────┐   │         │  ┌───────▼───────┐   │       │
│  │  │ Trainer       │   │         │  │ Optimizer     │   │       │
│  │  └───────┬───────┘   │         │  └───────┬───────┘   │       │
│  │  ┌───────▼───────┐   │         │  ┌───────▼───────┐   │       │
│  │  │ ModelRegistry │───┼─────────┼──│ Recommender   │   │       │
│  │  └───────────────┘   │         │  └───────────────┘   │       │
│  │  Outputs: *.pkl      │         │  In: query + graph   │       │
│  └─────────────────────┘         └─────────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│                     Feature extraction                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Query       │  │ Graph       │  │ Config      │              │
│  │ extractors  │  │ extractors  │  │ extractors  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Repository layout

```
autoconfig/
├── autoconfig/
│   ├── __init__.py
│   ├── cli.py
│   ├── feature_extractor/
│   │   ├── static_extractor.py
│   │   ├── symbolic_extractor.py
│   │   ├── graph_partition_extractor.py
│   │   ├── config_extractor.py
│   │   └── feature_manager.py
│   ├── offline/
│   │   ├── data_generator.py
│   │   ├── trainer.py
│   │   └── model_registry.py
│   ├── online/
│   │   ├── cost_predictor.py
│   │   ├── optimizer.py
│   │   └── recommender.py
│   └── models/
│       ├── bayesian_model.py
│       └── bayesian_models.py
├── data/
├── experiments/
├── examples/
├── tests/
└── docs/
```

## Modules

### 1. Feature extraction (`feature_extractor`)

**Static:** loops, branches, recursion, atomics, synchronization.

**Symbolic:** VScan, EScan, FScan, RExp, Atom, Comm templates (instantiated at merge time with graph stats).

**Graph:** vertex/edge counts, degree distribution, density, clustering, partition quality.

**Config:** CPU, memory, GPU, storage, and related scalars.

### 2. Offline (`offline`)

**DataGenerator** — synthetic training data (query templates, graph models, resource tuples).

**Trainer** — fits time and cost models from feature vectors.

**ModelRegistry** — versioning and metadata (active / deprecated).

### 3. Online (`online`)

**CostPredictor** — `predict_time`, `predict_cost`, `predict_both`, optional uncertainty.

**ConfigurationOptimizer** — rank candidates, top-K, perturb/refine.

**Recommender** — `recommend`, graph-specific helpers, what-if and compare utilities.

## Workflows

### Train

```bash
python -m autoconfig generate-data --n-samples 500 --output data/generated/
python -m autoconfig train --n-samples 500 --output data/models/
```

### Recommend

```bash
autoconfig recommend \
    --query my_algorithm.cu \
    --graph data/graph.csv \
    --top-n 3 \
    --output out/recommendation.yaml
```

### Python API (sketch)

```python
from autoconfig.offline.data_generator import DataGenerator
from autoconfig.offline.trainer import Trainer
from autoconfig.online.recommender import Recommender
from autoconfig.utils.query_complexity_extractor import QueryComplexityExtractor

generator = DataGenerator(seed=42)
dataset = generator.generate_dataset(num_samples=500)

trainer = Trainer("data/models/")
metrics = trainer.train(num_samples=500)

recommender = Recommender(model_dir="data/models/")
qc = QueryComplexityExtractor().extract_from_file("my_query.cu")
result = recommender.recommend_with_complexity(
    query_complexity=qc,
    graph_features={...},
    top_n=3,
)
```

## Data flow

Query code, graph data, and a config catalog are fed through extractors into a **combined feature vector** used for training; at inference time the same features drive **CostPredictor** and **Optimizer** to produce recommendations.

## Models

**Time model** — typical inputs include query-complexity scalars, graph statistics, and resource counts; output is predicted latency (often with intervals).

**Cost model** — same architecture; target is monetary or weighted cost derived from resources and duration.

## Extensibility

- New query **templates**: extend `DataGenerator.QUERY_TEMPLATES`.
- New **resource** shapes: extend `DataGenerator.RESOURCE_CONFIGS` or catalogs used by `ConfigGenerator`.
- New **features**: extend the relevant extractor and any merger / vector builder, then retrain.

## CLI summary

```bash
autoconfig query --input query.py --output out/query.yaml
autoconfig graph --input graph.csv --output out/graph.yaml
autoconfig config --num-samples 20 --output out/configs.yaml
autoconfig train --n-samples 500 --output data/models/
autoconfig recommend --query my_algorithm.cu --graph data/graph.csv --top-n 3
```

## API reference

See [api_reference.md](api_reference.md).
