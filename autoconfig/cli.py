#!/usr/bin/env python3
"""
AutoConfig CLI - Unified Command-Line Interface

Tools for extracting features from:
1. Query code
2. Graph data (edge list)
3. System configurations (LHS sampling)

Usage:
    autoconfig query --input query.py --output out/query.yaml
    autoconfig graph --input data/edges.csv --output out/graph.yaml
    autoconfig graph --input data/partitions/ --output out/graph.yaml
    autoconfig config --num-samples 20 --output out/configs.yaml
    autoconfig all --query query.py --graph data/ --config-n 20 --output out/
"""

import argparse
import sys
from pathlib import Path


def cmd_query(args):
    """Handle query feature extraction."""
    from .utils.query_feature_extractor import QueryFeatureExtractor
    
    extractor = QueryFeatureExtractor()
    features = extractor.extract_from_file(args.input)
    
    # Add metadata
    features['metadata'] = {
        'input_file': args.input,
        'output_file': args.output,
    }
    
    extractor.save_to_yaml(features, args.output)
    
    print(f"Query features extracted:")
    print(f"  Static: {features['feature_count']['static']}")
    print(f"  Symbolic: {features['feature_count']['symbolic']}")
    print(f"  Total: {features['feature_count']['total']}")
    print(f"  Output: {args.output}")
    print(f"  Note: Graph-dependent values use placeholders")


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
    print(f"  Balance: {gf['partition']['balance']:.4f}")
    print(f"  Output: {args.output}")


def cmd_config(args):
    """Handle configuration generation."""
    from .utils.config_generator import (
        ConfigGenerator,
        load_resource_catalog,
        generate_default_catalog,
        generate_simple_catalog,
    )
    from .utils.capacity_config_generator import (
        CapacityConstrainedConfigGenerator,
        load_capacity,
    )

    # Capacity-constrained generation
    if args.capacity:
        print(f"Loading system capacity: {args.capacity}")
        capacity = load_capacity(args.capacity)
        generator = CapacityConstrainedConfigGenerator(capacity)
        
        k_range = (args.k_min, args.k_max) if args.k_max else None
        print(f"\nGenerating {args.num_samples} configurations using LHS...")
        
        result = generator.generate(args.num_samples, k_range)
        generator.save_to_yaml(result, args.output)
        
        print(f"\nConfiguration generation complete:")
        print(f"  Generated: {len(result['configurations'])} configurations")
        print(f"  Max machines (by capacity): {result['metadata']['max_machines']}")
        print(f"  Output: {args.output}")
        
        # Show sample configurations
        print("\nSample configurations:")
        for i, config in enumerate(result['configurations'][:3]):
            r = config['resource']
            gpu_str = f", {r.get('num_gpus', 0)} GPU" if r.get('num_gpus', 0) > 0 else ""
            print(f"  Config {i}: k={config['k']} machines, "
                  f"CPU={r.get('cpu_cores', '?')} cores, "
                  f"Mem={r.get('memory_gb', '?')}GB{gpu_str}")
        return

    # Load or generate resource catalog
    if args.resource_catalog:
        print(f"Loading resource catalog: {args.resource_catalog}")
        catalog = load_resource_catalog(args.resource_catalog)
    elif args.cpu and args.memory:
        # Generate simple catalog from parameters
        print(f"Generating simple catalog from parameters:")
        print(f"  CPU: {args.cpu} cores")
        print(f"  Memory: {args.memory} GB")
        if args.gpu > 0:
            print(f"  GPU: {args.gpu} (memory: {args.gpu_memory} GB)")
        catalog = generate_simple_catalog(
            args.cpu,
            args.memory,
            args.gpu,
            args.gpu_memory,
            args.storage
        )
        print(f"  Created {len(catalog)} resource variations")
    elif args.use_default_catalog:
        print("Using default cloud-like resource catalog")
        catalog = generate_default_catalog()
    else:
        print("Error: Please provide one of:")
        print("  --capacity <file.yaml>  (system capacity)")
        print("  --resource-catalog <file.yaml>")
        print("  --cpu <cores> --memory <GB> [--gpu <num>] [--gpu-memory <GB>]")
        print("  --use-default-catalog")
        return

    print(f"Catalog size: {len(catalog)} resources")

    # Generate configurations
    generator = ConfigGenerator(catalog)
    k_range = (args.k_min, args.k_max if args.k_max else 16)

    print(f"\nGenerating {args.num_samples} configurations using LHS...")
    print(f"K range: {k_range}")

    result = generator.generate(args.num_samples, k_range)
    generator.save_to_yaml(result, args.output)

    print(f"\nConfiguration generation complete:")
    print(f"  Generated: {len(result['configurations'])} configurations")
    print(f"  Output: {args.output}")
    
    # Show sample configurations
    print("\nSample configurations:")
    for i, config in enumerate(result['configurations'][:3]):
        r = config['resource']
        gpu_str = f", GPU={r.get('num_gpus', 0)}" if r.get('num_gpus', 0) > 0 else ""
        print(f"  Config {i}: k={config['k']}, CPU={r.get('cpu_cores', '?')}, Mem={r.get('memory_gb', '?')}GB{gpu_str}")


def cmd_all(args):
    """Handle complete feature extraction pipeline."""
    from .utils.query_feature_extractor import QueryFeatureExtractor
    from .utils.graph_feature_extractor import GraphFeatureExtractor
    from .utils.config_generator import (
        ConfigGenerator,
        generate_default_catalog,
    )
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Complete Feature Extraction Pipeline")
    print("=" * 60)
    
    # 1. Query features
    if args.query:
        print("\n[1/3] Extracting query features...")
        query_extractor = QueryFeatureExtractor()
        query_features = query_extractor.extract_from_file(args.query)
        query_features['metadata'] = {'input_file': args.query}
        query_output = output_dir / 'query_features.yaml'
        query_extractor.save_to_yaml(query_features, str(query_output))
        print(f"  Query features: {query_features['feature_count']['total']} features")
        print(f"  Output: {query_output}")
    else:
        print("\n[1/3] Skipping query features (no input)")
    
    # 2. Graph features
    if args.graph:
        print("\n[2/3] Extracting graph features...")
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
        print(f"  Partitions: {gf['partition']['num_partitions']}")
        print(f"  Output: {graph_output}")
    else:
        print("\n[2/3] Skipping graph features (no input)")
    
    # 3. Configuration features
    print("\n[3/3] Generating configurations...")
    catalog = generate_default_catalog()
    generator = ConfigGenerator(catalog)
    k_range = (args.k_min, args.k_max)
    
    result = generator.generate(args.config_n, k_range)
    config_output = output_dir / 'config_features.yaml'
    generator.save_to_yaml(result, str(config_output))
    
    print(f"  Configurations: {len(result['configurations'])} samples")
    print(f"  Output: {config_output}")
    
    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


def cmd_merge(args):
    """Handle feature merging."""
    from .utils.feature_merger import FeatureMerger
    
    merger = FeatureMerger()
    result = merger.merge_all(args.query, args.graph, args.config)
    merger.save_to_yaml(result, args.output)
    
    print(f"Features merged successfully:")
    print(f"  Query: {args.query}")
    print(f"  Graph: {args.graph}")
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output}")
    print(f"  Feature matrix: {result['metadata']['num_samples']} samples × {result['metadata']['num_features']} features")


def main():
    parser = argparse.ArgumentParser(
        description='AutoConfig - Feature Extraction Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract query features
  autoconfig query --input query.py --output out/query.yaml

  # Extract graph features (single graph)
  autoconfig graph --input data/edges.csv --output out/graph.yaml

  # Extract graph features (partitioned graph)
  autoconfig graph --input data/partitions/ --output out/graph.yaml

  # Generate configurations
  autoconfig config --num-samples 20 --output out/configs.yaml --use-default-catalog

  # Complete pipeline
  autoconfig all --query query.py --graph data/ --config-n 20 --output out/
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
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Generate configurations using LHS')
    config_parser.add_argument('--resource-catalog', '-c', help='Resource catalog YAML file')
    config_parser.add_argument('--capacity', help='System capacity YAML file (capacity-constrained generation)')
    config_parser.add_argument('--num-samples', '-n', type=int, default=20, help='Number of samples')
    config_parser.add_argument('--k-min', type=int, default=1, help='Min instances')
    config_parser.add_argument('--k-max', type=int, default=None, help='Max instances (default: capacity-limited)')
    config_parser.add_argument('--output', '-o', default='out/config_features.yaml', help='Output YAML file')
    config_parser.add_argument('--use-default-catalog', action='store_true', help='Use default catalog')
    # Simple catalog parameters
    config_parser.add_argument('--cpu', type=int, default=None, help='CPU cores')
    config_parser.add_argument('--memory', type=int, default=None, help='Memory in GB')
    config_parser.add_argument('--gpu', type=int, default=0, help='Number of GPUs')
    config_parser.add_argument('--gpu-memory', type=int, default=0, help='GPU memory in GB')
    config_parser.add_argument('--storage', type=int, default=None, help='Storage in GB')
    config_parser.set_defaults(func=cmd_config)
    
    # All command
    all_parser = subparsers.add_parser('all', help='Complete feature extraction pipeline')
    all_parser.add_argument('--query', '-q', help='Query source file')
    all_parser.add_argument('--graph', '-g', help='Graph edge list file or folder')
    all_parser.add_argument('--config-n', type=int, default=20, help='Number of configurations')
    all_parser.add_argument('--k-min', type=int, default=1, help='Min instances')
    all_parser.add_argument('--k-max', type=int, default=16, help='Max instances')
    all_parser.add_argument('--output', '-o', default='out/', help='Output directory')
    all_parser.set_defaults(func=cmd_all)
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge query, graph, and config features')
    merge_parser.add_argument('--query', '-q', required=True, help='Query features YAML')
    merge_parser.add_argument('--graph', '-g', required=True, help='Graph features YAML')
    merge_parser.add_argument('--config', '-c', required=True, help='Config features YAML')
    merge_parser.add_argument('--output', '-o', default='out/merged_features.yaml', help='Output merged YAML')
    merge_parser.set_defaults(func=cmd_merge)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == '__main__':
    main()
