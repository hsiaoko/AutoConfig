#!/usr/bin/env python3
"""
AutoConfig CLI - Unified Command-Line Interface

Tools for:
1. Feature extraction (query, graph; config YAML is external)
2. Offline training (data generation, model training)
3. Online configuration recommendation
   (config candidate YAML for merge: author or generate externally; see feature_merger)

Usage:
    # Feature extraction
    autoconfig query --input query.py --output out/query.yaml
    autoconfig graph --input data/edges.csv --output out/graph.yaml

    # Offline training
    autoconfig train --n-samples 500 --output data/models/
    autoconfig train-merged --data-dir out/train --output out/models/
    autoconfig eval-merged --model out/models/bayesian_cost_merged.pkl --data-dir out/test
    autoconfig generate-data --n-samples 200 --output data/generated/

    # Online recommendation
    autoconfig recommend --query bfs --graph data/graph.csv --top-n 3
"""

import argparse
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


def cmd_graph(args):
    """Handle graph feature extraction."""
    from .utils.graph_feature_extractor import GraphFeatureExtractor

    extractor = GraphFeatureExtractor()
    input_path = Path(args.input)

    if input_path.is_file():
        print(f"Processing single graph: {args.input}")
        features = extractor.extract_single(args.input)
    elif input_path.is_dir():
        print(f"Processing partitioned graph: {args.input}")
        features = extractor.extract_partitioned(args.input)
    else:
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    extractor.save_to_yaml(features, args.output)

    gf = features['graph_features']
    print(f"Graph features extracted:")
    print(f"  Vertices: {gf['basic']['num_vertices']}")
    print(f"  Edges: {gf['basic']['num_edges']}")
    print(f"  Partitions: {gf['partition']['num_partitions']}")
    print(f"  Output: {args.output}")


def cmd_train(args):
    """Handle model training."""
    from .offline.trainer import Trainer

    print("=" * 60)
    print("Training Bayesian Models")
    print("=" * 60)

    trainer = Trainer(args.output)

    dataset_path = args.dataset if hasattr(args, 'dataset') else None
    metrics = trainer.train(
        dataset_path=dataset_path,
        num_samples=args.n_samples,
        verbose=True
    )

    print("\nTraining Summary:")
    print(f"  Time Model - Train R²: {metrics['time_model']['train_r2']:.4f}, Test R²: {metrics['time_model']['test_r2']:.4f}")
    print(f"  Cost Model - Train R²: {metrics['cost_model']['train_r2']:.4f}, Test R²: {metrics['cost_model']['test_r2']:.4f}")
    print(f"  Models saved to: {args.output}")


def cmd_train_merged(args):
    """Train Bayesian cost regressor on merged feature YAMLs (Y=cost, X=other features)."""
    from .offline.yaml_feature_trainer import train_bayesian_cost_from_merged_yamls

    print("=" * 60)
    print("Training Bayesian cost model (merged feature YAMLs)")
    print("=" * 60)

    train_bayesian_cost_from_merged_yamls(
        args.data_dir,
        args.output,
        test_split=args.test_split,
        seed=args.seed,
        pattern=args.pattern,
        model_basename=args.model_basename,
        exclude_static=args.exclude_static,
        exclude_symbolic=args.exclude_symbolic,
        verbose=True,
    )
    print(f"\nModel and metadata written under: {args.output}")


def cmd_eval_merged(args):
    """Evaluate a merged-feature cost model; print (and optional save) metrics."""
    import yaml
    from .offline.yaml_feature_trainer import evaluate_bayesian_cost_on_merged_dir

    result = evaluate_bayesian_cost_on_merged_dir(
        args.model,
        args.data_dir,
        pattern=args.pattern,
        check_feature_names=not args.no_check_feature_names,
    )
    m = result["metrics"]
    print("=" * 60)
    print("Eval (merged feature YAMLs, target = cost)")
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


def cmd_generate_data(args):
    """Handle synthetic data generation."""
    from .offline.data_generator import DataGenerator

    print("=" * 60)
    print("Generating Synthetic Dataset")
    print("=" * 60)

    generator = DataGenerator(seed=args.seed)

    print(f"\nGenerating {args.n_samples} samples...")
    dataset = generator.generate_dataset(
        num_samples=args.n_samples,
        graph_size_range=(args.graph_min, args.graph_max),
        edge_prob_range=(args.edge_prob_min, args.edge_prob_max),
    )

    generator.save_dataset(dataset, args.output, save_graphs=args.save_graphs)

    print(f"\nDataset generation complete:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Output: {args.output}")


def cmd_recommend(args):
    """Handle configuration recommendation."""
    from .online.recommender import Recommender
    from .utils.graph_feature_extractor import GraphFeatureExtractor
    from .utils.query_complexity_extractor import QueryComplexityExtractor

    print("=" * 60)
    print("Configuration Recommendation")
    print("=" * 60)

    # Initialize recommender
    model_dir = args.model_dir if hasattr(args, 'model_dir') and args.model_dir else None
    recommender = Recommender(model_dir=model_dir, auto_train=args.auto_train)

    if not recommender.models_loaded and not args.auto_train:
        print("\nWarning: No trained models found.")
        print("Using heuristic-based recommendation.")
        print("Run 'autoconfig train' first for better results.\n")

    # Load graph features
    print(f"Loading graph: {args.graph}")
    graph_extractor = GraphFeatureExtractor()
    graph_path = Path(args.graph)

    if graph_path.is_file():
        graph_features_dict = graph_extractor.extract_single(args.graph)
        graph_features = graph_features_dict['graph_features']
    elif graph_path.is_dir():
        graph_features_dict = graph_extractor.extract_partitioned(args.graph)
        graph_features = graph_features_dict['graph_features']
    else:
        print(f"Error: Graph path does not exist: {args.graph}")
        sys.exit(1)

    # Load query complexity from file
    print(f"Loading query: {args.query}")
    query_extractor = QueryComplexityExtractor()
    
    try:
        query_complexity = query_extractor.extract_from_file(args.query)
        print(f"  Query complexity: {query_complexity}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get recommendations
    print("\nRecommending configurations...")
    result = recommender.recommend_with_complexity(
        query_complexity=query_complexity,
        graph_features=graph_features,
        top_n=args.top_n
    )

    # Display results
    print("\n" + "=" * 60)
    print(f"Top {args.top_n} Diverse Recommendations:")
    print("=" * 60)

    for rec in result['recommendations']:
        print(f"\n[Rank {rec['rank']}] {rec['config_id']}")
        r = rec['resource']
        print(f"  CPU: {r.get('cpu_cores', '?')} cores")
        print(f"  Memory: {r.get('memory_gb', '?')} GB")
        print(f"  GPU: {r.get('num_gpus', 0)}")
        print(f"  Predicted Time: {rec['predicted_time_ms']:.2f} ms")
        print(f"  Predicted Cost: {rec['predicted_cost']:.4f}")
        if rec.get('is_perturbed'):
            print(f"  * Refined via perturbation")

    # Save results
    if args.output:
        recommender.save_recommendation(result, args.output)
        print(f"\nResults saved to: {args.output}")


def cmd_all(args):
    """Query + graph extraction only. Prepare config YAML separately for ``merge``."""
    from .utils.query_feature_extractor import QueryFeatureExtractor
    from .utils.graph_feature_extractor import GraphFeatureExtractor

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Query & graph feature extraction (config step removed from pipeline)")
    print("=" * 60)

    # 1. Query features
    if args.query:
        print("\n[1/2] Extracting query features...")
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
    else:
        print("\n[1/2] Skipping query features")

    # 2. Graph features
    if args.graph:
        print("\n[2/2] Extracting graph features...")
        graph_extractor = GraphFeatureExtractor()
        graph_path = Path(args.graph)

        if graph_path.is_file():
            graph_features = graph_extractor.extract_single(args.graph)
        else:
            graph_features = graph_extractor.extract_partitioned(args.graph)

        graph_output = output_dir / 'graph_features.yaml'
        graph_extractor.save_to_yaml(graph_features, str(graph_output))

        gf = graph_features['graph_features']
        print(f"  Graph: {gf['basic']['num_vertices']} vertices, {gf['basic']['num_edges']} edges")
    else:
        print("\n[2/2] Skipping graph features")

    print("\n  Config candidates: not generated here. Author a config YAML, then run:")
    print("  autoconfig merge -q <query> -g <graph> -c <config.yaml> -o <merged.yaml>")

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
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
        description='AutoConfig - Graph Query Configuration System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Feature extraction
  autoconfig query --input query.py --output out/query.yaml
  autoconfig graph --input graph.csv --output out/graph.yaml
  autoconfig merge -q out/query.yaml -g out/graph.yaml -c my_config.yaml -o out/merged.yaml

  # Training
  autoconfig train --n-samples 500 --output data/models/
  autoconfig train-merged --data-dir out/train --output out/models/
  autoconfig eval-merged -m out/models/bayesian_cost_merged.pkl -d out/test
  autoconfig generate-data --n-samples 200 --output data/generated/

  # Recommendation
  autoconfig recommend --query bfs --graph data/graph.csv --top-n 3
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Query command
    query_parser = subparsers.add_parser('query', help='Extract query code features')
    query_parser.add_argument('--input', '-i', required=True, help='Input query source file')
    query_parser.add_argument('--output', '-o', default='out/query_features.yaml', help='Output YAML file')
    query_parser.set_defaults(func=cmd_query)

    # Graph command
    graph_parser = subparsers.add_parser('graph', help='Extract graph data features')
    graph_parser.add_argument('--input', '-i', required=True, help='Input edge list file or folder')
    graph_parser.add_argument('--output', '-o', default='out/graph_features.yaml', help='Output YAML file')
    graph_parser.set_defaults(func=cmd_graph)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train Bayesian models')
    train_parser.add_argument('--n-samples', type=int, default=500, help='Number of training samples')
    train_parser.add_argument('--dataset', type=str, default=None, help='Existing dataset path')
    train_parser.add_argument('--output', '-o', default='data/models/', help='Output directory for models')
    train_parser.set_defaults(func=cmd_train)

    train_merged_parser = subparsers.add_parser(
        'train-merged',
        help='Train Bayesian cost model on merged YAMLs (target = cost, inputs = other features)',
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
        '--exclude-static',
        action='store_true',
        help='Drop static program features (static_*) from merged X',
    )
    train_merged_parser.add_argument(
        '--exclude-symbolic',
        action='store_true',
        help='Drop graph-parameterized symbolic features (sym_*) from merged X',
    )
    train_merged_parser.set_defaults(func=cmd_train_merged)

    eval_merged_parser = subparsers.add_parser(
        "eval-merged",
        help="Evaluate Bayesian cost model on a folder of merged feature YAMLs",
    )
    eval_merged_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to trained .pkl (e.g. out/models/bayesian_cost_merged.pkl)",
    )
    eval_merged_parser.add_argument(
        "--data-dir",
        "-d",
        required=True,
        help="Folder with merged feature YAMLs (ground-truth cost in first feature slot)",
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
    eval_merged_parser.set_defaults(func=cmd_eval_merged)

    # Generate data command
    gen_parser = subparsers.add_parser('generate-data', help='Generate synthetic training data')
    gen_parser.add_argument('--n-samples', type=int, default=200, help='Number of samples')
    gen_parser.add_argument('--graph-min', type=int, default=100, help='Min graph vertices')
    gen_parser.add_argument('--graph-max', type=int, default=5000, help='Max graph vertices')
    gen_parser.add_argument('--edge-prob-min', type=float, default=0.001, help='Min edge probability')
    gen_parser.add_argument('--edge-prob-max', type=float, default=0.05, help='Max edge probability')
    gen_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    gen_parser.add_argument('--save-graphs', action='store_true', help='Save graphs as edge lists')
    gen_parser.add_argument('--output', '-o', default='data/generated/', help='Output directory')
    gen_parser.set_defaults(func=cmd_generate_data)

    # Recommend command
    rec_parser = subparsers.add_parser('recommend', help='Recommend optimal configuration')
    rec_parser.add_argument('--query', '-q', required=True,
                           help='Query source file (.cu, .cpp, .py)')
    rec_parser.add_argument('--graph', '-g', required=True, help='Graph file or directory')
    rec_parser.add_argument('--top-n', type=int, default=3, help='Number of recommendations')
    rec_parser.add_argument('--model-dir', '-m', default=None, help='Model directory')
    rec_parser.add_argument('--auto-train', action='store_true', help='Auto-train if no models')
    rec_parser.add_argument('--output', '-o', default=None, help='Save results to YAML')
    rec_parser.set_defaults(func=cmd_recommend)

    # All command
    all_parser = subparsers.add_parser('all', help='Extract query and graph features (no config gen)')
    all_parser.add_argument('--query', '-q', help='Query source file')
    all_parser.add_argument('--graph', '-g', help='Graph edge list file or folder')
    all_parser.add_argument('--output', '-o', default='out/', help='Output directory')
    all_parser.set_defaults(func=cmd_all)

    # Merge command
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
