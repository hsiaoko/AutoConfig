# Documentation index

| Document | Description |
|----------|-------------|
| [FEATURE_PIPELINE.md](FEATURE_PIPELINE.md) | **Four-step YAML pipeline** (query → graph → config → merge) |
| [feature_extraction.md](feature_extraction.md) | Static / symbolic / graph / config features; merged **53-D** layout and link to training (**X** = dims 4–53, **Y** = **`-y` `0|1|2`**) |
| [CLI_GUIDE.md](CLI_GUIDE.md) | `autoconfig` CLI commands |
| [usage_guide.md](usage_guide.md) | Usage guide (workflow, Python API, **§ Training merged YAMLs (53-D)**) |
| [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | Config YAML for merge; LHS helper `data/conf/build_ten_conf.py` |
| [api_reference.md](api_reference.md) | Classes and functions |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture |
| [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md) | **Train / test** on merged **53-d** YAMLs: `train-merged` / `eval-merged` — **X = 4–53**; **Y = -y 0|1|2**; **`--model-kind`** (`bayesian` / `nn` / `mlp` / `rl` stub); `merge_abc_features.py` → `fill_merged_train_costs_from_stats.py` |

The repository **[README.md](../README.md)** at the project root has installation, quick start, and links here.

### Merged training (53-D) — quick reference

- **Row layout:** `price`, `time`, `cost` (3) + static (8) + symbolic (12) + graph/partition (17) + config (13) = **53**. Same order as :class:`~autoconfig.utils.feature_merger.FeatureMerger` (`conf_price` is the last config slot).
- **What the regressor uses:** **input X** = dimensions **4–53** (0-based index `3` onward). **Label** = one of the first three slots, chosen on the CLI with **`-y N`** / **`--y-axis N`** where **N ∈ {0,1,2}** (table: [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md#label-y-in-the-cli--y----y-axis)).
- **Read next:** [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md) (options, `*_meta.yaml`, `eval-merged`). CLI entry points: [CLI_GUIDE.md](CLI_GUIDE.md) § *Training*.
