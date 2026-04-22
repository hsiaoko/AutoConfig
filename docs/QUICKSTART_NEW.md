# AutoConfig quick usage (modules overview)

## Install

```bash
cd /path/to/AutoConfig
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Modules

1. **feature_extractor/** — query, graph, and config features  
2. **offline/** — synthetic data and training  
3. **online/** — cost prediction and configuration recommendation  

## Train

```bash
autoconfig train --n-samples 500 --output data/models/
```

Optional synthetic data:

```bash
autoconfig generate-data --n-samples 200 --output data/generated/
```

## Recommend (query **file** + graph)

```bash
autoconfig recommend \
    --query my_query.cu \
    --graph data/graph.csv \
    --top-n 3 \
    --output out/recommendation.yaml
```

Supported sources include `.cu`, `.cpp`, `.py`. The harness estimates complexity signals such as `v_scan`, `e_scan`, `f_scan`, `atomic`, and `sync`.

## Python sketch

```python
from autoconfig.offline.data_generator import DataGenerator
from autoconfig.offline.trainer import Trainer
from autoconfig.online.recommender import Recommender

generator = DataGenerator(seed=42)
dataset = generator.generate_dataset(num_samples=500)
generator.save_dataset(dataset, "data/generated/")

trainer = Trainer("data/models/")
trainer.train(num_samples=500)

recommender = Recommender(model_dir="data/models/")
graph_features = {
    "num_vertices": 10000,
    "num_edges": 50000,
    "avg_degree": 5.0,
    "max_degree": 100,
    "density": 0.001,
    "avg_clustering": 0.5,
}
result = recommender.recommend(
    query_name="pagerank",
    graph_features=graph_features,
    top_n=3,
)
```

## Feature extraction CLI

```bash
autoconfig query --input query.py --output out/query.yaml
autoconfig graph --input graph.csv --output out/graph.yaml
# python data/conf/build_ten_conf.py -n 10   # or author configs (CONFIG_GUIDE.md)
autoconfig all --query q.py --graph g.csv --output out/
```

## Troubleshooting

**No trained models:** run `autoconfig train` first or pass `--auto-train` to `recommend` where supported.

**Import errors:** activate the venv and `pip install -e .`.

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md)  
- [quickstart.md](quickstart.md)  
- [CLI_GUIDE.md](CLI_GUIDE.md)  
