# Merged features: training and evaluation

This document describes how to **train** a Bayesian cost regressor and **evaluate** it on a held-out folder, using the **merged feature YAML** format produced by `autoconfig merge` (or equivalent tooling). Paths below are **examples**; all CLI arguments accept your own locations.

**Related reading:** [FEATURE_PIPELINE.md](FEATURE_PIPELINE.md) (merge step), [feature_extraction.md](feature_extraction.md) (vector layout), [CLI_GUIDE.md](CLI_GUIDE.md) (all `autoconfig` commands).

---

## Data layout

Each training or test item is one YAML file with:

- `feature_names`: a list of length 51. The **first** name must be `cost` (regression target). The other 50 names are **inputs** to the model, in a fixed order shared across all files in a run.
- `feature_vector`: same length as `feature_names`, aligned by index. The first value is the **ground-truth** cost; the next 50 values are the feature row **X** (same order as the merge pipeline).

All files in a directory must use the **same** `feature_names` list (in the same order). Typical layout:

- **Train directory:** e.g. `out/train/`, a glob of `*.yaml` (or a narrower `pattern` such as `*_gridgraph_wcc.yaml`).
- **Test directory:** the same format, e.g. `out/test/`, for reporting MAE, RMSE, MAPE, and R² on unseen examples.

Filling the `cost` slot for training often comes from measured runs (e.g. timings from a benchmark); for evaluation, `cost` is still required so the metrics compare prediction to truth.

**Bulk fill from GridGraph `STATS.txt`:** run `scripts/fill_merged_train_costs_from_stats.py` (from the repo). It matches `out/features/conf_XX_<data>_gridgraph_<task>.yaml` to `../GridGraph/<data>_gridgraph_<task>/STATS.txt` (overridable with `--gridgraph-root`), and writes the same filenames under `out/train/`. Workloads with no `STATS.txt` are skipped; use `--strict` to fail instead.

---

## Environment

The `autoconfig` entry point is available after a local install. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

If `autoconfig: command not found`, either activate the venv or run:

```bash
.venv/bin/python -m autoconfig.cli <subcommand> ...
```

Helper scripts under `scripts/` (see below) call the venv the same way.

---

## Training: `train-merged`

Fits a **Bayesian** cost model (`BayesianCostModel` in the codebase: log target, normalized inputs, variational fit). It writes:

- `bayesian_cost_merged.pkl` (or a name you choose) — the trained model
- `bayesian_cost_merged_meta.yaml` — training metadata, including `feature_names_x`, and train/test metrics when a split is used

**Basic usage:**

```bash
autoconfig train-merged --data-dir out/train --output out/models
```

**Useful options:**

| Option | Description |
|--------|-------------|
| `--data-dir`, `-d` | Directory of merged feature YAMLs (required) |
| `--output`, `-o` | Where to write `.pkl` and `*_meta.yaml` (default `out/models`) |
| `--test-split` | Fraction of files reserved for a random test split, for printed metrics (default `0.2`). Use `0` to train on **all** files (no internal test) |
| `--seed` | RNG seed for the split (default `42`) |
| `--pattern` | Glob under `data-dir` (default `*.yaml`), e.g. `*_gridgraph_wcc.yaml` to train on one kernel family only |
| `--model-basename` | Basename for output files (default `bayesian_cost_merged`) |
| `--exclude-static` | Drop static program features (names starting with `static_`), i.e. SPF |
| `--exclude-symbolic` | Drop graph-parameterized symbolic features (names starting with `sym_`), i.e. SGF |

**Ablation scripts** (convenience wrappers; they set a distinct `--model-basename` and pass `--data-dir out/train --output out/models --test-split 0` when you run them with no arguments):

| Script | Effect | Default output basename |
|--------|--------|-------------------------|
| `scripts/run_train_merged_no_spf.sh` | `--exclude-static` | `bayesian_cost_merged_no_spf` |
| `scripts/run_train_merged_no_sgf.sh` | `--exclude-symbolic` | `bayesian_cost_merged_no_sgf` |
| `scripts/run_train_merged_no_pf.sh` | both flags (only **graph** + **config** columns) | `bayesian_cost_merged_no_pf` |

**Shell helper** (default data/output relative to repo root; override by passing the same flags yourself):

```bash
chmod +x scripts/run_train_merged.sh
./scripts/run_train_merged.sh
# or
./scripts/run_train_merged.sh --data-dir out/train --output out/models --test-split 0
```

---

## Evaluation (test): `eval-merged`

Loads a trained `.pkl` and a **test directory** of merged YAMLs, predicts `cost`, and compares to the `cost` field in each file. Prints aggregate metrics; optionally writes a YAML report and lists per-file errors.

**Basic usage:**

```bash
autoconfig eval-merged \
  --model out/models/bayesian_cost_merged.pkl \
  --data-dir out/test
```

**Options:**

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Path to the `.pkl` from `train-merged` (required) |
| `--data-dir`, `-d` | Test folder of merged YAMLs (required) |
| `--pattern` | Same as training (default `*.yaml`) |
| `--output`, `-o` | Optional path to write the full result dict (metrics + `per_file` list) as YAML |
| `--show-per-file` | Also print one line per file: `y_true`, `y_pred`, `abs_error` |
| `--no-check-feature-names` | Do not require `feature_names_x` to match the model meta (dimension must still match). Use only if you know the layout is equivalent |

A `*_meta.yaml` next to the model is used, when present, to apply the same **feature exclusions** as training (`exclude_static` / `exclude_symbolic` are read from the meta), then to verify that the filtered `feature_names_x` and dimension match the model. You do not need to pass extra flags to `eval-merged` for ablation models: use the correct `.pkl` and its sidecar metadata.

**Shell helper:**

```bash
chmod +x scripts/run_eval_merged.sh
./scripts/run_eval_merged.sh -m out/models/bayesian_cost_merged.pkl -d out/test
./scripts/run_eval_merged.sh -m out/models/bayesian_cost_merged.pkl -d out/test -o out/eval_report.yaml
```

---

## Metrics reported

For evaluation, the **same** definitions are used as in training-side reporting:

- **MAE** — mean absolute error
- **RMSE** — root mean square error
- **MAPE (%)** — mean absolute percentage error, with a small floor on `y_true` to avoid division by zero
- **R²** — coefficient of determination relative to the mean of `y` on the evaluated set

---

## Python API

For scripts or notebooks:

```python
from pathlib import Path
from autoconfig.offline.yaml_feature_trainer import (
    load_merged_feature_dir,
    train_bayesian_cost_from_merged_yamls,
    evaluate_bayesian_cost_on_merged_dir,
)

# Train (example paths)
train_bayesian_cost_from_merged_yamls(
    Path("out/train"),
    Path("out/models"),
    test_split=0.2,
    pattern="*.yaml",
    model_basename="bayesian_cost_merged",
    verbose=True,
)

# Evaluate
result = evaluate_bayesian_cost_on_merged_dir(
    "out/models/bayesian_cost_merged.pkl",
    "out/test",
    pattern="*.yaml",
    check_feature_names=True,
)
print(result["metrics"])
# result["per_file"] — list of {file, y_true, y_pred, abs_error}
```

---

## Differences from `autoconfig train`

The default **`autoconfig train`** subcommand uses the **legacy offline pipeline** (synthetic `DataGenerator` samples and a different feature shape). The **`train-merged` / `eval-merged`** flow is the one to use for **real merged 51×1 vectors** (1 target + 50 features) from the YAML merge pipeline.

---

## See also

- [CLI_GUIDE.md](CLI_GUIDE.md) — install and all commands
- [docs/README.md](README.md) — documentation index
