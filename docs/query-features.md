# Query features

Extract **`query_features`** from a query **source file**: static scalars (loops, branches, atomics, sync, …) and **format_version: 2** symbolic families (templates instantiated with graph stats at merge — see [feature-merge.md](feature-merge.md)).

There is **no** `autoconfig graph` subcommand; graph-feature YAML must be prepared separately.

## CLI: `autoconfig query`

```bash
autoconfig query -h
```

| Argument | Short | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--input` | `-i` | yes | — | Path to query source (language depends on parser support; see examples under `data/queries/`). |
| `--output` | `-o` | no | `out/query_features.yaml` | Output query YAML. |

## CLI: `autoconfig all`

Writes only **`query_features.yaml`** under the output directory (batch-friendly); you still supply graph and config for merge yourself.

```bash
autoconfig all -h
```

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--query` | `-q` | yes | — | Query source path. |
| `--output` | `-o` | no | `out/` | Output directory (created); writes `<dir>/query_features.yaml`. |

## Related scripts

There is **no dedicated** `scripts/*.py` wrapper for query extraction; loop over files with shell + installed `autoconfig`.

Optional helper:

| Script | Note |
|--------|------|
| `scripts/generate_sample_graphs.py` | Library-style edge-list CSV generators (Erdős–Rényi, …); **no argparse main**; does not emit query YAML. |

## Next steps

1. Prepare `graph_features.yaml` (nested structure must match `FeatureMerger`; see library examples).
2. Prepare config YAML ([conf-generation.md](conf-generation.md)).
3. Run `autoconfig merge` or `scripts/merge_abc_features.py` ([feature-merge.md](feature-merge.md)).
