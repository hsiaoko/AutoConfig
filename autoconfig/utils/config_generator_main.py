def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Generate candidate configurations using Latin Hypercube Sampling'
    )
    parser.add_argument(
        '--resource-catalog', '-c',
        type=str,
        default=None,
        help='Path to resource catalog YAML file'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=20,
        help='Number of configurations to generate'
    )
    parser.add_argument(
        '--k-min',
        type=int,
        default=1,
        help='Minimum number of instances'
    )
    parser.add_argument(
        '--k-max',
        type=int,
        default=16,
        help='Maximum number of instances'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='out/config_features.yaml',
        help='Output YAML file path'
    )
    parser.add_argument(
        '--use-default-catalog',
        action='store_true',
        help='Use default cloud-like resource catalog'
    )
    # Simple catalog parameters
    parser.add_argument(
        '--cpu',
        type=int,
        default=None,
        help='CPU cores (creates simple catalog with 3 variations)'
    )
    parser.add_argument(
        '--memory',
        type=int,
        default=None,
        help='Memory in GB'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='Number of GPUs'
    )
    parser.add_argument(
        '--gpu-memory',
        type=int,
        default=0,
        help='GPU memory in GB'
    )
    parser.add_argument(
        '--storage',
        type=int,
        default=None,
        help='Storage in GB (default: 10x memory)'
    )
    
    args = parser.parse_args()
    
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
        print("  --resource-catalog <file.yaml>")
        print("  --cpu <cores> --memory <GB> [--gpu <num>] [--gpu-memory <GB>]")
        print("  --use-default-catalog")
        print("\nExamples:")
        print("  autoconfig config -n 10 -o out/config.yaml --cpu 16 --memory 64")
        print("  autoconfig config -n 10 -o out/config.yaml --cpu 32 --memory 128 --gpu 4 --gpu-memory 32")
        print("  autoconfig config -n 20 -o out/config.yaml --use-default-catalog")
        return
    
    print(f"Catalog size: {len(catalog)} resources")
    
    # Create generator
    generator = ConfigGenerator(catalog)
    
    # Generate configurations
    k_range = (args.k_min, args.k_max)
    print(f"\nGenerating {args.num_samples} configurations using LHS...")
    print(f"K range: {k_range}")
    
    result = generator.generate(args.num_samples, k_range)
    
    # Save to YAML
    generator.save_to_yaml(result, args.output)
    
    # Print summary
    print(f"\nConfiguration generation complete:")
    print(f"  Generated: {len(result['configurations'])} configurations")
    print(f"  K range: {result['metadata']['k_range']}")
    print(f"  Output: {args.output}")
    
    # Show first few configurations
    print("\nSample configurations:")
    for i, config in enumerate(result['configurations'][:3]):
        r = config['resource']
        gpu_str = f", GPU={r.get('num_gpus', 0)}" if r.get('num_gpus', 0) > 0 else ""
        print(f"  Config {i}: k={config['k']}, "
              f"CPU={r.get('cpu_cores', '?')} cores, "
              f"Mem={r.get('memory_gb', '?')}GB{gpu_str}")


if __name__ == '__main__':
    main()
