# Feature YAML pipeline (four steps)

This guide matches the library entry points `autoconfig query|graph|merge` (and config generation via `data/conf/build_ten_conf.py` or hand-authored YAML).

## Data flow

```text
Query source          Edge list(s)           Resource catalog
      │                     │                        │
      ▼                     ▼                        ▼
 step 1                  step 2                   step 3
query_features.yaml   graph_features.yaml    config_features.yaml
      │                     │                        │
      └─────────────────────┴────────────────────────┘
                              │
                              ▼
                         step 4
                    merged_features.yaml
              (numeric matrix + feature names)
```

- **Step 1** — `Φ_static`: real-valued code-structure features. `Φ_sym`: six symbolic **families** with `template`, `formula`, `detected`, and `requires_*` lists (`format_version: 2`). No graph numbers here.
- **Step 2** — Graph and (optional) partition statistics used to **instantiate** symbolic templates in step 4.
- **Step 3** — Candidate configurations (e.g. Latin Hypercube Sampling over a resource catalog). YAML lists `configurations`.
- **Step 4** — `FeatureMerger` loads the three YAML files, evaluates symbolic templates into **12 numeric** symbolic features, concatenates static, graph, and per-configuration features, and writes `feature_matrix` plus `feature_names`.

## Commands

From the **repository root** (ensure the `autoconfig` package is installed: `pip install -e .`).

### Step 1 — Query features

```bash
autoconfig query --input path/to/query.py --output out/query_features.yaml
```

### Step 2 — Graph features

This package does **not** provide a CLI to build `graph_features.yaml` from edge lists. Produce that file with your own pipeline (schema: [feature_extraction.md](feature_extraction.md)) so **merge** can read it.

### Step 3 — System configuration

Author `out/config_features.yaml`, or generate candidate files with:

```bash
python data/conf/build_ten_conf.py -n 10
```

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for schema, GPU/grid constraints, and CLI options.

### Step 4 — Merge

```bash
autoconfig merge \
  --query out/query_features.yaml \
  --graph out/graph_features.yaml \
  --config out/config_features.yaml \
  --output out/merged_features.yaml
```

## Key implementation files

| Concern | Location |
|--------|----------|
| Query static + symbolic YAML | `autoconfig/utils/query_feature_extractor.py` |
| Symbolic templates (`extract_symbolic_expressions`) | `autoconfig/feature_extractor/symbolic_extractor.py` |
| Graph YAML for merge | **You provide** `graph_features.yaml` (see [feature_extraction.md](feature_extraction.md)) |
| Config YAML + LHS samples | `data/conf/build_ten_conf.py`, [CONFIG_GUIDE.md](CONFIG_GUIDE.md) |
| Merge + symbolic instantiation | `autoconfig/utils/feature_merger.py` |

## Symbolic instantiation rules (summary)

`FeatureMerger.graph_stats_for_symbolic_instantiation()` maps `graph_features.yaml` into statistics named in the paper (e.g. `num_vertices`, `diameter`, `boundary_degree_sum`). Partition count `num_partitions` is used as **n** in \|V\|/n and \|E\|/n when present. Undetected template families yield zero `sym_*_coeff` and `sym_*_requires`.

For the full feature list and formulas, see [feature_extraction.md](feature_extraction.md).
