#!/usr/bin/env python3
"""
AutoConfig CLI — feature extraction, merge, merged-YAML training / evaluation.

Stages are composed from library tools in :mod:`autoconfig.utils`, :mod:`autoconfig.feature_extractor`,
and :mod:`autoconfig.merged`. Config YAML generation (e.g. LHS) lives under ``data/conf/`` (separate scripts).

Usage:
    # Query feature extraction (graph_features.yaml: provide separately, see docs)
    autoconfig query --input query.py --output out/query.yaml

    # Merged training (53-D YAMLs)
    autoconfig train-merged --data-dir out/train --output out/models/
    autoconfig eval-merged -m out/models/bayesian_cost_merged.pkl -d out/test
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_query(args):
    """Handle query feature extraction."""
    from .utils.query_feature_extractor import QueryFeatureExtractor

    extractor = QueryFeatureExtractor()
    features = extractor.extract_from_file(args.input)

    features['metadata'] = {
        'input_file': args.input,
        'output_file': args.output,
    }

    extractor.save_to_yaml(features, args.output)

    fc = features['feature_count']
    print(f"Query features extracted:")
    print(f"  Static scalars: {fc['static']}")
    print(f"  Symbolic template families: {fc.get('symbolic_families', fc.get('symbolic', '?'))}")
    print(f"  Numeric symbolic dims after merge: {fc.get('symbolic_numeric_dim_after_merge', '?')}")
    print(f"  Output: {args.output}")


def cmd_train_merged(args):
    """Train a merged-feature regressor (default Bayesian); Y from --y-axis; X = cols 4..end."""
    from .merged import MERGED_Y_AXIS_NAMES, train_bayesian_cost_from_merged_yamls
    from .models.merged_registry import available_model_kinds

    y_axis: int = int(args.y_axis)
    y_name = MERGED_Y_AXIS_NAMES[y_axis]
    mk = (args.model_kind or "bayesian").strip().lower()
    model_options = None
    if getattr(args, "model_options", None):
        model_options = json.loads(args.model_options)
        if not isinstance(model_options, dict):
            raise SystemExit("--model-options must be a JSON object, e.g. '{\"max_iter\": 300}'")
    print("=" * 60)
    print(
        f"Training merged regressor (kind={mk!r}, y_axis={y_axis} → {y_name!r}; "
        f"X = features 4..end)"
    )
    print("=" * 60)
    if getattr(args, "batch_size", None) is not None:
        print(f"conf_batch_size override (all rows): {args.batch_size}")
    if getattr(args, "graph_e_only", False):
        print("graph features: |E| only (graph_num_edges); |V| dropped")

    ve_only: bool | None = None
    if getattr(args, "graph_ve_only", False):
        ve_only = True
    if getattr(args, "full_graph_with_no_pf", False):
        ve_only = False

    kinds = [k.lower() for k in available_model_kinds()]
    if mk not in kinds:
        raise SystemExit(
            f"Unknown --model-kind {args.model_kind!r}. One of: {', '.join(sorted(set(kinds)))}"
        )
    train_bayesian_cost_from_merged_yamls(
        args.data_dir,
        args.output,
        test_split=args.test_split,
        seed=args.seed,
        pattern=args.pattern,
        model_basename=args.model_basename,
        target_name=y_name,
        y_axis=y_axis,
        n_iter=args.n_iter,
        exclude_static=args.exclude_static,
        exclude_symbolic=args.exclude_symbolic,
        ve_only_graph=ve_only,
        graph_e_only=getattr(args, "graph_e_only", False),
        conf_batch_size=getattr(args, "batch_size", None),
        model_kind=mk,
        model_options=model_options,
        verbose=True,
    )
    print(f"\nModel and metadata written under: {args.output}")


def cmd_eval_merged(args):
    """Evaluate a merged-feature model; print (and optional save) metrics."""
    import yaml
    from .merged import (
        MERGED_Y_AXIS_NAMES,
        evaluate_bayesian_cost_on_merged_dir,
        optional_meta_for_model,
    )

    y_axis: int | None = args.y_axis
    meta0 = optional_meta_for_model(Path(args.model))
    if y_axis is None:
        if meta0 and meta0.get("y_axis") is not None:
            y_axis = int(meta0["y_axis"])
        elif meta0 and str(meta0.get("target", "")) in MERGED_Y_AXIS_NAMES:
            y_axis = MERGED_Y_AXIS_NAMES.index(str(meta0["target"]))
        else:
            y_axis = 1
    y_name = MERGED_Y_AXIS_NAMES[int(y_axis)]

    result = evaluate_bayesian_cost_on_merged_dir(
        args.model,
        args.data_dir,
        pattern=args.pattern,
        target_name=y_name,
        y_axis=int(y_axis),
        check_feature_names=not args.no_check_feature_names,
        conf_batch_size=getattr(args, "batch_size", None),
    )
    m = result["metrics"]
    y_note = f"y_axis={int(y_axis)} → {y_name!r}"
    print("=" * 60)
    print(f"Eval (merged feature YAMLs, {y_note})")
    print("=" * 60)
    print(f"  model:      {result['model_path']}")
    print(f"  data_dir:   {result['data_dir']}")
    print(f"  pattern:    {result['pattern']}")
    print(f"  n_samples:  {result['n_samples']}")
    print("  --- metrics ---")
    print(f"  MAE:        {m['mae']:.6f}")
    print(f"  RMSE:       {m['rmse']:.6f}")
    print(f"  MAPE (%):   {m['mape']:.4f}")
    print(f"  R²:         {m['r2']:.6f}")
    if args.show_per_file:
        print("  --- per file ---")
        for row in result["per_file"]:
            print(
                f"  {row['y_true']:.4f}  pred={row['y_pred']:.4f}  "
                f"abs_err={row['abs_error']:.4f}  {row['file']}"
            )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
        print(f"  Full result written to: {out_path}")


def cmd_all(args):
    """Write query features under output dir. Supply ``graph_features.yaml`` and config YAML yourself before merge."""
    from .utils.query_feature_extractor import QueryFeatureExtractor

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Query feature extraction (graph/config YAML: separate)")
    print("=" * 60)

    if not args.query:
        print("Error: --query / -q is required for ``all``.", file=sys.stderr)
        sys.exit(1)

    print("\nExtracting query features...")
    query_extractor = QueryFeatureExtractor()
    query_features = query_extractor.extract_from_file(args.query)
    query_output = output_dir / 'query_features.yaml'
    query_extractor.save_to_yaml(query_features, str(query_output))
    qfc = query_features['feature_count']
    print(
        f"  Query: {qfc['static']} static + "
        f"{qfc.get('symbolic_families', '?')} symbolic families "
        f"(→ {qfc.get('symbolic_numeric_dim_after_merge', '?')} numeric slots after merge)"
    )
    print(f"  Wrote: {query_output}")

    print("\n  Add ``graph_features.yaml`` and config YAML, then:")
    print("  autoconfig merge -q <query.yaml> -g <graph.yaml> -c <config.yaml> -o <merged.yaml>")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"Output directory: {output_dir}")


def cmd_merge(args):
    """Handle feature merging."""
    from .utils.feature_merger import FeatureMerger

    merger = FeatureMerger()
    result = merger.merge_all(args.query, args.graph, args.config)
    merger.save_to_yaml(result, args.output)

    print(f"Features merged successfully:")
    print(f"  Output: {args.output}")
    print(f"  Feature matrix: {result['metadata']['num_samples']} samples × {result['metadata']['num_features']} features")


def main():
    parser = argparse.ArgumentParser(
        description='AutoConfig — graph query features, merge, merged-model train/eval',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  autoconfig query --input query.py --output out/query.yaml
  autoconfig merge -q out/query.yaml -g out/graph.yaml -c my_config.yaml -o out/merged.yaml
  autoconfig train-merged --data-dir out/train --output out/models/
  autoconfig eval-merged -m out/models/bayesian_cost_merged.pkl -d out/test
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    query_parser = subparsers.add_parser('query', help='Extract query code features')
    query_parser.add_argument('--input', '-i', required=True, help='Input query source file')
    query_parser.add_argument('--output', '-o', default='out/query_features.yaml', help='Output YAML file')
    query_parser.set_defaults(func=cmd_query)

    train_merged_parser = subparsers.add_parser(
        'train-merged',
        help='Train on merged 53-d YAMLs: X = features 4..53; Y = --y-axis (0|1|2, see docs)',
    )
    train_merged_parser.add_argument(
        '-y',
        '--y-axis',
        type=int,
        default=1,
        choices=(0, 1, 2),
        metavar='N',
        help='Which label to predict: 0=price, 1=time, 2=cost (default: 1)',
    )
    train_merged_parser.add_argument(
        '--data-dir', '-d', required=True,
        help='Directory of merged feature YAML files (e.g. out/train)',
    )
    train_merged_parser.add_argument(
        '--output', '-o', default='out/models',
        help='Directory to save model pickle and metadata YAML',
    )
    train_merged_parser.add_argument(
        '--test-split', type=float, default=0.2,
        help='Fraction held out for test metrics (0 = train on all)',
    )
    train_merged_parser.add_argument('--seed', type=int, default=42, help='Shuffle seed for split')
    train_merged_parser.add_argument(
        '--pattern', type=str, default='*.yaml',
        help='Glob under data-dir (e.g. "*_gridgraph_wcc.yaml")',
    )
    train_merged_parser.add_argument(
        '--model-basename', type=str, default='bayesian_cost_merged',
        help='Basename for .pkl and _meta.yaml files',
    )
    train_merged_parser.add_argument(
        '--n-iter', type=int, default=300,
        help='For model-kind=bayesian: max variational iterations. Ignored for other kinds unless set in --model-options',
    )
    train_merged_parser.add_argument(
        '--model-kind',
        type=str,
        default='bayesian',
        metavar='NAME',
        help='Regressor: bayesian (default), nn or mlp (neural net), rl (placeholder until you register a backend)',
    )
    train_merged_parser.add_argument(
        '--model-options',
        type=str,
        default=None,
        metavar='JSON',
        help='JSON kwargs for the chosen backend, e.g. '
        '\'{"hidden_layer_sizes":[128,64],"max_iter":400,"early_stopping":false}\' for nn/mlp',
    )
    train_merged_parser.add_argument(
        '--exclude-static',
        action='store_true',
        help='Drop static program features (static_*) from merged X',
    )
    train_merged_parser.add_argument(
        '--exclude-symbolic',
        action='store_true',
        help='Drop graph-parameterized symbolic features (sym_*) from merged X',
    )
    train_merged_parser.add_argument(
        '--full-graph-with-no-pf',
        action='store_true',
        help='With --exclude-static and --exclude-symbolic, keep all graph_* and partition_* '
        "columns (legacy no_pf). Default without this flag: only graph_num_vertices, "
        'graph_num_edges, and conf_*.',
    )
    train_merged_parser.add_argument(
        '--graph-ve-only',
        action='store_true',
        help='no_graph ablation: drop all graph_/partition_ except graph_num_vertices and '
        'graph_num_edges; keep static_*, sym_*, and conf_*. In meta, ve_only_graph=true with '
        'exclude_static/symbolic=false. Overridden by --full-graph-with-no-pf.',
    )
    train_merged_parser.add_argument(
        '--graph-e-only',
        action='store_true',
        help='Among graph_/partition_ keep only graph_num_edges (|E|). Implies a minimal graph '
        'block; use with --exclude-static --exclude-symbolic and --graph-ve-only for no_all: '
        'X = conf_* + |E| only.',
    )
    train_merged_parser.add_argument(
        '--batch-size',
        type=float,
        default=None,
        metavar='N',
        help='Override merged feature conf_batch_size to N for every sample (I/O batch in '
        "config, not SGD). Recorded in *_meta.yaml as conf_batch_size_override.",
    )
    train_merged_parser.set_defaults(func=cmd_train_merged)

    eval_merged_parser = subparsers.add_parser(
        "eval-merged",
        help="Evaluate a train-merged model (Bayesian, mlp, …) on a folder of merged feature YAMLs",
    )
    eval_merged_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to trained .pkl (e.g. out/models/bayesian_cost_merged.pkl)",
    )
    eval_merged_parser.add_argument(
        "-y",
        "--y-axis",
        type=int,
        default=None,
        choices=(0, 1, 2),
        metavar="N",
        help="Override label: 0=price, 1=time, 2=cost. If omitted, use *_meta.yaml, else 1",
    )
    eval_merged_parser.add_argument(
        "--data-dir",
        "-d",
        required=True,
        help="Folder with merged feature YAMLs (Y slot per -y / --y-axis or model meta)",
    )
    eval_merged_parser.add_argument(
        "--pattern",
        type=str,
        default="*.yaml",
        help="Glob under data-dir (same as training)",
    )
    eval_merged_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write full result (metrics + per_file) to this YAML",
    )
    eval_merged_parser.add_argument(
        "--show-per-file",
        action="store_true",
        help="Print y_true, y_pred, abs_error for each file",
    )
    eval_merged_parser.add_argument(
        "--no-check-feature-names",
        action="store_true",
        help="Do not require feature_names to match model meta (dim must still match)",
    )
    eval_merged_parser.add_argument(
        "--batch-size",
        type=float,
        default=None,
        metavar="N",
        help="Set conf_batch_size in every test row to N before predict (use same N as training "
        "if you overrode batch at train time; omit to use each YAML's value).",
    )
    eval_merged_parser.set_defaults(func=cmd_eval_merged)

    all_parser = subparsers.add_parser(
        'all',
        help='Write query_features.yaml in output dir (requires --query; graph/config YAML separate)',
    )
    all_parser.add_argument('--query', '-q', required=True, help='Query source file')
    all_parser.add_argument('--output', '-o', default='out/', help='Output directory')
    all_parser.set_defaults(func=cmd_all)

    merge_parser = subparsers.add_parser('merge', help='Merge features')
    merge_parser.add_argument('--query', '-q', required=True, help='Query features YAML')
    merge_parser.add_argument('--graph', '-g', required=True, help='Graph features YAML')
    merge_parser.add_argument('--config', '-c', required=True, help='Config features YAML')
    merge_parser.add_argument('--output', '-o', default='out/merged_features.yaml', help='Output YAML')
    merge_parser.set_defaults(func=cmd_merge)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()
