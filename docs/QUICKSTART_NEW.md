# AutoConfig quick usage (modules overview)

## Install

```bash
cd /path/to/AutoConfig
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Modules

1. **feature_extractor/**, **utils/** — query extraction, **merge** (graph YAML you provide)  
2. **merged/** — 53-D YAML `train-merged` / `eval-merged`, `MergedBayesianPredictor`  
3. **prediction/** (optional) — `CostPredictor` for `FeatureManager` + code/graph/config (`from autoconfig.prediction import CostPredictor`)  
4. **data/conf/** — e.g. `build_ten_conf.py` for LHS config YAMLs (standalone script)  

## Train (merged)

```bash
autoconfig train-merged --data-dir out/train --output out/models
autoconfig eval-merged -m out/models/bayesian_cost_merged.pkl -d out/test
```

## Python sketch

```python
from autoconfig.merged import train_bayesian_cost_from_merged_yamls, load_merged_feature_dir
from autoconfig import MergedBayesianPredictor

train_bayesian_cost_from_merged_yamls("out/train", "out/models", y_axis=1, test_split=0.2, verbose=True)

pred = MergedBayesianPredictor.load("out/models/bayesian_cost_merged.pkl")
# pred.predict_batch(X)  # rows aligned to meta["feature_names_x"]
# pred.predict_merged_doc(doc)  # one YAML object
```

## Feature extraction CLI

```bash
autoconfig query --input query.py --output out/query.yaml
# graph_features.yaml: external
# python data/conf/build_ten_conf.py -n 10
autoconfig all --query q.py --output out/
```

## Troubleshooting

**Import errors:** `pip install -e .` from repo root with venv activated.

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md)  
- [quickstart.md](quickstart.md)  
- [CLI_GUIDE.md](CLI_GUIDE.md)  
