# Documentation

The repo uses a **high-level → topic** split: the root [README.md](../README.md) states goals and the pipeline; this folder holds stage-specific guides. **Each guide lists related `scripts/` and every CLI flag with defaults and allowed values.**

| Topic | File |
|--------|------|
| System config YAML (LHS / grid) | [conf-generation.md](conf-generation.md) |
| Query program features | [query-features.md](query-features.md) |
| Feature merge & batch merge | [feature-merge.md](feature-merge.md) |
| Training & evaluation on merged YAMLs | [training.md](training.md) |
| Config recommendation with a trained model | [conf-recommend.md](conf-recommend.md) |

Suggested reading order follows the data path: config generation → query features → (graph features you provide) → merge → optional time/cost labeling → train → recommend.
