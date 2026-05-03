# AutoConfig

Build numeric features from static and symbolic program traits, graph and partition statistics, and system configuration (resources and CUDA launch parameters); train regressors to predict graph-query **price / time / cost**; **rank** candidate configurations for a task.

## Overview: data flow

1. **Config YAML**: one file per configuration (or many files under a directory) with node `resource` and `catalog` / `configurations`.
2. **Query-feature YAML**: extracted from query source — `static` and `symbolic` templates (instantiated at merge time).
3. **Graph-feature YAML**: |V|, degree stats, diameter, partition / boundary metrics (there is **no** `autoconfig graph`; supply your own or external tooling).
4. **Merge**: `FeatureMerger` produces one **53-D** row: `price`, `time`, `cost` (first three = label slots) + eight `static_*` + twelve `sym_*` + seventeen graph/partition scalars + thirteen `conf_*`. Training uses **X = dimensions 4 through end**; **Y** is chosen with **`-y` / `--y-axis`** in **`0|1|2`** (price / time / cost).
5. **Train & recommend**: `train-merged` on a directory of merged YAMLs; `recommend-conf` for a new query + graph + candidate pool.

Step-by-step notes, script entry points, and **full CLI parameter lists** live under `docs/` (index: `docs/readme.md`).

## Install

```bash
cd AutoConfig
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Use the `autoconfig` entry point (same as `python -m autoconfig.cli`).

## Command cheat sheet

```bash
# Query features
autoconfig query -i data/queries/gridgraph_bfs.cpp -o out/query_features.yaml

# Single triple merge (one YAML each)
autoconfig merge -q out/query_features.yaml -g out/graph_features.yaml \
  -c data/conf/conf_01.yaml -o out/merged_features.yaml

# Train (default target is time → y=1)
autoconfig train-merged -d out/train -o out/models --model-basename my_run

# Evaluate
autoconfig eval-merged -m out/models/my_run.pkl -d out/test

# Config recommendation
autoconfig recommend-conf -m out/models/my_run.pkl \
  -q out/query_features.yaml -g out/graph_features.yaml -c data/conf/gpu/ -k 5
```

## Documentation

| Doc | Topic |
|-----|--------|
| [docs/readme.md](docs/readme.md) | Index |
| [docs/conf-generation.md](docs/conf-generation.md) | Config YAML sampling & scripts |
| [docs/query-features.md](docs/query-features.md) | Query feature extraction |
| [docs/feature-merge.md](docs/feature-merge.md) | Merge & A×B×C batch merge |
| [docs/training.md](docs/training.md) | Train, eval, labels, ablation wrappers |
| [docs/conf-recommend.md](docs/conf-recommend.md) | Config recommendation & smoke tests |

## Repository layout (excerpt)

```
autoconfig/          # library & CLI (cli.py)
data/conf/           # config generators & sample output dirs
data/queries/        # sample query sources
scripts/             # batch, LHS/grid configs, label fill, train/recommend wrappers
docs/                # topic docs above
```

## License

MIT License
