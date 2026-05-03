# System configuration (conf) generation

Merge expects **merge-style config YAML**: top-level `catalog`, `configurations` (at least one `node_id` + `resource`), optional `metadata`. This repo batch-generates **one configuration per file** via **Latin Hypercube (LHS)** or a **product grid** on an 8-D unit cube, and can optionally call Perplexity to fill cloud-pricing metadata.

## Scripts

| Script | Role |
|--------|------|
| `data/conf/build_ten_conf.py` | Writes `conf_NN.yaml` under `--out-dir` (default: `data/conf/`); LHS + GPU branch rules; optional pricing API. |
| `scripts/generate_confs.py` | **Required** output dir `-o`; `--method lhs|grid`; shares `_build_resource` rules with `build_ten_conf`. |
| `scripts/generate_confs.sh` | Wrapper: prefers `.venv/bin/python scripts/generate_confs.py`, else `python3`; forwards **all** `"$@"`. |

## `data/conf/build_ten_conf.py`

From the repo root:

```bash
python data/conf/build_ten_conf.py -h
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-n`, `--num` | int | `10` | Number of files (≥ 1). |
| `--seed` | int | `42` | LHS RNG seed. |
| `--out-dir` | Path | `data/conf/` (script dir) | Output directory. |
| `--prefix` | str | `conf` | Filename prefix, e.g. `conf_01.yaml`. |
| `--gpu-fraction` | float | `0.5` | Must be in **[0,1]**: LHS 4th axis controls CPU-only vs GPU branch share (higher → more GPU rows). |
| `--cpu-min`, `--cpu-max` | int | `1`, `64` | Discrete range for `cpu_cores`. |
| `--mem-min`, `--mem-max` | int | `1`, `128` | `memory_gb`. |
| `--storage-min`, `--storage-max` | int | `32`, `4096` | `storage_gb`. |
| `--grid-min`, `--grid-max` | int | `1`, `256` | `grid_size` (GPU branch). |
| `--block-min`, `--block-max` | int | `1`, `256` | `block_size`. |
| `--num-gpus-min`, `--num-gpus-max` | int | `1`, `8` | `num_gpus` (GPU branch). |
| `--gpu-mem-min`, `--gpu-mem-max` | int | `8`, `64` | `gpu_memory_gb`. |
| `--sm-base` | int | `2560` | Per-GPU SM multiplier: `gpu_sm_count = sm_base * num_gpus`. |
| `--with-cloud-pricing` | flag | off | If an API key is available, fill `metadata.cloud_pricing`. |
| `--perplexity-api-key` | str | `""` | Override `PERPLEXITY_API_KEY`. |
| `--perplexity-model` | str | `""` | Override `PERPLEXITY_MODEL` (default sonar). |
| `--cloud-pricing-delay` | float | `1.0` | Sleep seconds between pricing calls (rate limiting). |

### Post-generation constraints

- `num_gpus == 0` ⇒ `grid_size == 0` and `block_size == 0`.
- `num_gpus > 0` ⇒ `grid_size > 0` and `block_size > 0`.

## `scripts/generate_confs.py` / `scripts/generate_confs.sh`

```bash
./scripts/generate_confs.sh -h
# or
.venv/bin/python scripts/generate_confs.py -h
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-n`, `--n-conf` | int | **required** | Number of files (≥ 1). |
| `-o`, `--output` | Path | **required** | Output directory (created if missing). |
| `--method` | choice | `lhs` | **`lhs`**: 8-D LHS; **`grid`**: product grid in the unit cube (first `n` tuples in row-major index order). |
| `--seed` | int | `42` | LHS seed; `grid` uses `_unit_cube_rows_grid` (deterministic). |
| `--prefix` | str | `conf` | `{prefix}_01.yaml`. |
| `--gpu-fraction` | float | `0.5` | **[0,1]**, same semantics as `build_ten_conf._build_resource`. |
| `--cpu-min` … `--gpu-mem-max`, `--sm-base` | int | see below | **Defaults differ from `build_ten_conf`**: here `cpu-max=128`, `grid/block max=128`, `num-gpus` default **0–1**. |

Quick default ranges for `generate_confs.py`: `--cpu-max 128`, `--mem-max 128`, `--storage 32–4096`, `--grid-max 128`, `--block-max 128`, `--num-gpus-min 0`, `--num-gpus-max 1`, `--gpu-mem 8–64`, `--sm-base 2560`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--with-cloud-pricing` | flag | off | Same as `build_ten_conf` when a key is present. |
| `--perplexity-api-key` | str | `""` | Same. |
| `--perplexity-model` | str | `""` | Same. |
| `--cloud-pricing-delay` | float | `1.0` | Same. |

`generate_confs.sh` does not parse flags itself; it passes **`"$@"`** to `generate_confs.py`.

## Next steps

- Single-file merge: `autoconfig merge -c <conf.yaml>` (see [feature-merge.md](feature-merge.md)).
- Recommendation with a directory of candidates: one `*.yaml` per configuration (see [conf-recommend.md](conf-recommend.md)).
- Batch GPU benchmarks: `scripts/run_exp_conf_gpu_benchmarks.sh` (env-driven; see the “Experiments” section in [training.md](training.md)).
