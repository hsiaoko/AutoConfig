# Documentation index

| Document | Description |
|----------|-------------|
| [FEATURE_PIPELINE.md](FEATURE_PIPELINE.md) | **Four-step YAML pipeline** (query → graph → config → merge) |
| [feature_extraction.md](feature_extraction.md) | Static / symbolic / graph / config features; merged **53-D** layout and link to training (**X** = dims 4–53, **Y** = `--y-axis` / `--target`) |
| [CLI_GUIDE.md](CLI_GUIDE.md) | `autoconfig` CLI commands |
| [usage_guide.md](usage_guide.md) | Usage guide (workflow, Python API, **§ Training merged YAMLs (53-D)**) |
| [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | Config YAML for merge; LHS helper `data/conf/build_ten_conf.py` |
| [api_reference.md](api_reference.md) | Classes and functions |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture |
| [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md) | **Train / test** on merged **53-d** YAMLs: `autoconfig train-merged` / `eval-merged` — **X = features 4–53**; **Y =** `price` / `time` / `cost` via **`--y-axis`** (0/1/2) or **`--target`**; shell helpers, metrics, pipeline `merge_abc_features.py` → `fill_merged_train_costs_from_stats.py` |

The repository **[README.md](../README.md)** at the project root has installation, quick start, and links here.

### Merged training (53-D) — quick reference

- **Row layout:** `price`, `time`, `cost` (3) + static (8) + symbolic (12) + graph/partition (17) + config (13) = **53**. Same order as :class:`~autoconfig.utils.feature_merger.FeatureMerger` (`conf_price` is the last config slot).
- **What the regressor uses:** **input X** = dimensions **4–53** (0-based index `3` onward) — only program / graph / config features. **Label Y** = exactly one of dimensions **1–3**, chosen with **`--y-axis 0|1|2`** (price / time / cost) or **`--target price|time|cost`**; if both are given, `--y-axis` wins.
- **Read next:** [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md) (options, `*_meta.yaml`, `eval-merged`). CLI entry points: [CLI_GUIDE.md](CLI_GUIDE.md) § *Training and recommendation*.
