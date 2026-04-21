# AutoConfig — Graph query execution time prediction

Feature extraction and machine learning for estimating graph query **runtime and cost**, using a four-stage pipeline: **static features**, **symbolic templates (formulas)**, **graph and partition statistics**, and **system configuration** (as in the paper).

---

## Installation

At the repository root (where `pyproject.toml` / `requirements.txt` live):

```bash
cd AutoConfig
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Then use the `autoconfig` CLI, or run the four pipeline steps with `python experiments/scripts/step*.py`.

---

## Feature pipeline (four YAML stages)

Aligned with the paper (§5): **query YAML** (numeric static + symbolic templates) → **graph YAML** → **system config YAML** → **merged numeric feature matrix**.

| Step | Role | Output file |
|------|------|-------------|
| 1 | Extract `Φ_static` and symbolic templates `Φ_sym` from query source (`format_version: 2`, formulas; not final graph coefficients) | `query_features.yaml` |
| 2 | Graph and partition stats from an edge list (or shard directory) | `graph_features.yaml` |
| 3 | LHS (etc.) sampling of candidate configs | `config_features.yaml` |
| 4 | **Instantiate** symbolic templates with graph stats; concatenate static / graph / config | `merged_features.yaml` |

### Option A: Four standalone scripts (best for following the paper)

From the repo root:

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

### Option B: `autoconfig` subcommands

```bash
autoconfig query  --input data/queries/kernel_bfs.cu --output out/query_features.yaml
autoconfig graph  --input data/edges.csv --output out/graph_features.yaml
autoconfig config --num-samples 20 --output out/config_features.yaml
autoconfig merge   --query out/query_features.yaml --graph out/graph_features.yaml \
                   --config out/config_features.yaml --output out/merged_features.yaml
```

Generate the first three artifacts in one shot (merge is separate):

```bash
autoconfig all --query query.cu --graph data/ --config-n 20 --output out/
```

For richer config generation (`--use-default-catalog`, `--resource-catalog`, `--cpu` / `--memory`, etc.), use `python experiments/scripts/step3_system_config.py --help`.

---

## Symbolic features

- In **step 1** YAML, `symbolic` is the **template layer**: each pattern (VScan, EScan, FScan, RExp, Atom, Comm) includes `template`, `formula`, `detected`, and required graph/partition fields—not final scalars such as `|V|`, `|E|`.
- **Step 4** reads `graph_features.yaml`; `FeatureMerger` turns templates into **12** numeric features (`sym_*_coeff` / `sym_*_requires`) for downstream models.
- Legacy flat `sym_*` placeholders in query YAML are still supported.

Formulas and field tables: **[docs/feature_extraction.md](docs/feature_extraction.md)**. Pipeline details: **[docs/FEATURE_PIPELINE.md](docs/FEATURE_PIPELINE.md)**.

---

## Documentation index

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/FEATURE_PIPELINE.md](docs/FEATURE_PIPELINE.md) | Four-step YAML pipeline |
| [docs/feature_extraction.md](docs/feature_extraction.md) | Feature definitions and merged layout |
| [docs/usage_guide.md](docs/usage_guide.md) | Usage guide (Python API and workflow) |
| [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md) | Configuration and resource catalog |
| [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) | CLI reference |
| [docs/api_reference.md](docs/api_reference.md) | API reference |

---

## Capabilities

### Query features

- **Static**: loop depth, branches, variables, recursion, atomics, sync, explicit parallelism, etc.
- **Symbolic**: six workload templates + formula strings; evaluated at merge time using graph stats.

### Graph features

Single edge-list CSV or **partition directory** (multiple files): vertices/edges, degree distribution, diameter, clustering, partition and boundary metrics (see `graph_features.yaml`).

### Configuration generation

Latin Hypercube sampling over a resource catalog yields `(k, resource)` candidates in `config_features.yaml`.

---

## Project layout (excerpt)

```
AutoConfig/
├── autoconfig/
│   ├── cli.py
│   ├── feature_extractor/
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
├── docs/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Training and recommendation (brief)

```bash
autoconfig train --n-samples 500 --output data/models/
```

```bash
autoconfig recommend --query my_algorithm.cu --graph data/graph.csv --top-n 3 --output out/recommendation.yaml
```

More options and experiment scripts: [experiments/README.md](experiments/README.md).

---

## Multi-agent development harness (optional)

See [dev_harness/QUICKSTART.md](dev_harness/QUICKSTART.md).

---

## License

MIT License
