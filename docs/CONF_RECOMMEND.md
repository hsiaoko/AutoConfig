# Configuration recommendation (`recommend-conf`)

Use a **trained `train-merged` regressor** (Bayesian, NN/MLP, or any registered `MergedTabularRegressor`) to **rank configuration candidates** for **one fixed query (task)** and **one fixed graph**: for each candidate configuration, AutoConfig builds the same **53-D merged row** as in training (via `FeatureMerger`), predicts the **target** the model was trained on (`price` / `time` / `cost`, from `*_meta.yaml`), and returns the **top-k** by predicted value.

**Related:** training layout [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md), merge semantics [FEATURE_PIPELINE.md](FEATURE_PIPELINE.md), config file shape [CONFIG_GUIDE.md](CONFIG_GUIDE.md).

---

## Prerequisites

1. **Model:** path to **`<stem>.pkl`** from `train-merged`, with sibling **`<stem>_meta.yaml`** (required). The loader (`MergedBayesianPredictor.load`) reads **`feature_names_x`**, **`y_axis`**, **`exclude_static` / `exclude_symbolic`**, **`ve_only_graph`**, **`graph_e_only`**, optional **`conf_batch_size_override`**, and applies the **same** preprocessing as `eval-merged`.
2. **Query YAML:** same structure as training-time query features (top-level `query_features` block), e.g. from `autoconfig query`.
3. **Graph YAML:** same structure as training-time graph features (`graph_features` block).
4. **Config candidates:** must be compatible with `FeatureMerger.extract_config_features` (see below).

Predictions are **purely model-based**; validate promising configs with real measurements when possible.

---

## CLI: `autoconfig recommend-conf`

```bash
autoconfig recommend-conf \
  -m <model.pkl> \
  -q <query_features.yaml> \
  -g <graph_features.yaml> \
  -c <config.yaml_or_directory> \
  [-k N] [-o DIR_OR_FILE] [options...]
```

### Options

| Option | Required | Description |
|--------|------------|-------------|
| `-m`, `--model` | yes | Path to `train-merged` **`.pkl`**; **`<stem>_meta.yaml`** must sit beside it. |
| `-q`, `--query` | yes | Query (task) features YAML. |
| `-g`, `--graph` | yes | Graph features YAML. |
| `-c`, `--config` | yes | **Single YAML** (supports multiple rows in `configurations:`) **or** a **directory** of `*.yaml` / `*.yml` (see [Config input shapes](#config-input-shapes)). |
| `-k`, `--k` | no | Top **N** configs (default **5**). Clamped to the number of candidates. |
| `--maximize` | no | Rank by **largest** predicted value (default: **minimize**, e.g. lower cost/time). |
| `--batch-size` | no | Override merged **`conf_batch_size`** on every row (match training if you used `--batch-size` there). |
| `-o`, `--output` | no | **`Directory`:** write one YAML per ranked row (`recommend_rank01.yaml`, …). **`Path ending in `.yaml`/`.yml`:** write a **single** summary file (`ConfRecommendResult.to_dict`). |
| `--export-prefix` | no | Basename prefix under `-o` directory (default **`recommend`** → `recommend_rank01.yaml`). |
| `--full-report` | no | With `-o` as directory, also write **`{prefix}_full_report.yaml`**. |
| `--refine` | no | After initial top-k, **local search**: random perturbations of CPU/memory/GPU grid-block (see [Local refinement](#local-refinement)). |
| `--refine-max-iterations` | no | Cap refine rounds (default **10000**). |
| `--refine-seed` | no | RNG seed for perturbations. |

The **predicted target name** is taken from model meta (`target` / `y_axis`), not from the CLI.

---

## Config input shapes

### A) Directory of YAMLs (typical for GPU conf grids)

`-c /path/to/dir/` — non-recursive: **only top-level** `*.yaml` / `*.yml`, sorted by filename.

- Each file may look like **`data/conf/gpu/conf_01.yaml`**: `catalog` + `configurations:` with **one** row (or several).
- Or a **bare** configuration object (must include at least **`resource`** or **`node_id`** fields recognizable by the loader).

The **first** file that defines **`catalog`** supplies pricing/catalog for feature extraction (unless you only have defaults). Candidates are concatenated in sorted file order; each recommendation item may record **`source_path`** back to the originating file.

Implementation: `autoconfig.conf_recommend.recommend._load_config_dir`.

### B) Single YAML

`-c candidates.yaml` — standard **`configurations:`** list (and optional **`catalog`**), same idea as `autoconfig merge -c`.

---

## Outputs

### Stdout

The CLI prints human-readable ranks: predicted value, optional `source_path`, and a YAML dump of each **`configuration`** object.

### `-o` as directory

`autoconfig.conf_recommend.export_configs.export_recommendations_to_dir` writes **`metadata`** (rank, predicted, model path, target, optional refine stats), **`catalog`**, and **`configurations:`** with **one** row per file — ready to reuse in **`merge`** or deployment tooling.

### `-o` as `*.yaml` file

Single document = `ConfRecommendResult.to_dict()` (all ranked items + catalog + refine metadata).

---

## Local refinement (`--refine`)

Optional second phase (`autoconfig.conf_recommend.refine_search`):

- Starts from the initial top-**k** **configurations**.
- Each iteration: **perturb** each parent (`perturb_configuration`): `cpu_cores` ±16 (clamped ≥1), `memory_gb` in ±16 GB steps (up to ±48 GB), and if `num_gpus` > 0 randomly **×2 or ÷2** `grid_size` / `block_size` (minimum 1).
- Re-merge and re-predict children; update global best; continue until an iteration yields **no strict improvement** vs current best, or **`--refine-max-iterations`** is hit.

Refined results store **`initial_items`**, **`refine_iterations`**, **`refine_stopped_reason`** on `ConfRecommendResult`. Exported YAML metadata may mirror these fields.

---

## Shell helpers

| Script | Role |
|--------|------|
| `scripts/run_recommend_conf.sh` | Thin wrapper: forwards all args to `autoconfig recommend-conf` (requires venv / `autoconfig` on PATH). |
| `scripts/run_recommend_conf_dir.sh` | Same as above when you pass flags; **with no args**, reads **`MERGED_MODEL`**, **`QUERY_YAML`**, **`GRAPH_YAML`**, **`CONF_DIR`**, **`K`**, optional **`RECOMMEND_EXPORT_DIR`** (auto `-o`). |

Smoke tests (synthetic train + recommend):

- `scripts/recommend_conf_smoke.sh`
- `scripts/recommend_conf_smoke_dir.sh`

---

## Python API

```python
from pathlib import Path
from autoconfig.conf_recommend import recommend_top_k, export_recommendations_to_dir

r = recommend_top_k(
    model_path="out/models/my_run.pkl",
    query_features=Path("out/query_features/kernel_bfs.yaml"),
    graph_features=Path("out/graph_features/friendster.yaml"),
    config_candidates=Path("data/conf/gpu"),  # or single YAML path / dict / list[dict]
    k=5,
    lower_is_better=True,
    conf_batch_size=None,
    local_refine=False,
)
paths = export_recommendations_to_dir(r, Path("out/recommended_confs"), prefix="recommend")
```

Core types: `ConfRecommendation`, `ConfRecommendResult` in `autoconfig.conf_recommend.recommend`.

---

## Consistency checklist

- **Same ablation as training:** if the model was trained with `--exclude-symbolic` / `--graph-ve-only` / …, **meta** tells the predictor which columns to keep — your **query** and **graph** YAMLs should still contain the fields the merger expects (dropped columns are zeroed or skipped per merger logic).
- **Same `conf_batch_size` override:** if training used `--batch-size`, pass **`--batch-size`** here too for aligned predictions.
- **Catalog / price:** if training used real catalog prices in config YAMLs, ensure candidate configs use a compatible **`catalog`** so **`price`** / **`conf_price`** slots match intent.
