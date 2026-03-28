"""
Configuration Generator using Latin Hypercube Sampling (LHS)

Generates candidate configurations by sampling from resource catalog.

Usage:
    python -m autoconfig.utils.config_generator \
        --resource-catalog resources.yaml \
        --num-samples 20 \
        --output out/configs.yaml
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from scipy.stats import qmc


class ConfigGenerator:
    """
    Generate candidate configurations using Latin Hypercube Sampling.
    
    Based on the paper's method:
    - Sample from resource catalog using LHS over major hardware dimensions
    - Generate compact candidate set with broad coverage
    - Each configuration = (k instances, resource_type)
    """
    
    def __init__(self, resource_catalog: List[Dict[str, Any]]):
        """
        Initialize with resource catalog.
        
        Args:
            resource_catalog: List of resource descriptions
                Each resource has:
                - cpu_cores: int
                - memory_gb: int
                - storage_gb: int
                - num_gpus: int
                - gpu_sm_count: int (optional)
                - gpu_memory_gb: int (optional)
        """
        self.catalog = resource_catalog
        self.n_resources = len(resource_catalog)
        
        # Extract catalog statistics
        self.catalog_stats = self._compute_catalog_stats()
    
    def _compute_catalog_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute min/max/mean for each resource dimension."""
        if not self.catalog:
            return {}
        
        dimensions = ['cpu_cores', 'memory_gb', 'storage_gb', 'num_gpus']
        optional_dims = ['gpu_sm_count', 'gpu_memory_gb']
        
        # Add optional dimensions if present
        if any('gpu_sm_count' in r for r in self.catalog):
            dimensions.extend(optional_dims)
        
        stats = {}
        
        for dim in dimensions:
            values = [r.get(dim, 0) for r in self.catalog]
            stats[dim] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
            }
        
        return stats
    
    def generate_lhs_samples(
        self,
        num_samples: int,
        k_range: tuple = (1, 16)
    ) -> List[Dict[str, Any]]:
        """
        Generate configurations using Latin Hypercube Sampling.
        
        Args:
            num_samples: Number of configurations to generate
            k_range: (min_k, max_k) range for number of instances
            
        Returns:
            List of configuration dictionaries
        """
        if not self.catalog:
            raise ValueError("Resource catalog is empty")
        
        # Dimensions for LHS
        dimensions = ['cpu_cores', 'memory_gb', 'storage_gb', 'num_gpus']
        optional_dims = ['gpu_sm_count', 'gpu_memory_gb']
        
        # Add optional dimensions if present in catalog
        if any(dim in r for r in self.catalog for dim in optional_dims):
            dimensions.extend([d for d in optional_dims if any(d in r for r in self.catalog)])
        
        n_dims = len(dimensions)
        
        # Generate LHS samples in [0, 1]^n_dims
        sampler = qmc.LatinHypercube(d=n_dims, seed=42)
        samples = sampler.random(n=num_samples)
        
        # Scale samples to catalog ranges
        configurations = []
        
        for i in range(num_samples):
            sample = samples[i]
            
            # Find closest resource in catalog
            best_resource = None
            best_distance = float('inf')
            
            for resource in self.catalog:
                # Compute normalized distance
                distance = 0
                for j, dim in enumerate(dimensions):
                    if dim in resource:
                        dim_range = self.catalog_stats[dim]['max'] - self.catalog_stats[dim]['min']
                        if dim_range > 0:
                            normalized_value = (resource[dim] - self.catalog_stats[dim]['min']) / dim_range
                            distance += (sample[j] - normalized_value) ** 2
                
                distance = np.sqrt(distance)
                
                if distance < best_distance:
                    best_distance = distance
                    best_resource = resource
            
            # Sample k (number of instances)
            k = int(np.round(
                k_range[0] + sample.mean() * (k_range[1] - k_range[0])
            ))
            k = max(k_range[0], min(k_range[1], k))
            
            # Create configuration
            config = {
                'k': k,
                'resource': best_resource.copy() if best_resource else self.catalog[0],
                'config_id': i,
            }
            
            configurations.append(config)
        
        # Remove duplicates (same resource and k)
        seen = set()
        unique_configs = []
        
        for config in configurations:
            key = (config['k'], tuple(sorted(config['resource'].items())))
            if key not in seen:
                seen.add(key)
                unique_configs.append(config)
        
        # If we lost too many samples, fill with variations
        while len(unique_configs) < num_samples:
            idx = len(unique_configs)
            base_config = configurations[idx % len(configurations)].copy()
            base_config['config_id'] = idx
            base_config['k'] = max(1, base_config['k'] + np.random.randint(-1, 2))
            unique_configs.append(base_config)
        
        return unique_configs[:num_samples]
    
    def generate_config_features(
        self,
        configurations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract features from configurations.
        
        Args:
            configurations: List of configuration dictionaries
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for config in configurations:
            k = config['k']
            resource = config['resource']
            
            # Configuration features
            features = {
                'conf_k_instances': k,
                'conf_total_cpu_cores': k * resource.get('cpu_cores', 1),
                'conf_total_memory_gb': k * resource.get('memory_gb', 8),
                'conf_total_storage_gb': k * resource.get('storage_gb', 100),
                'conf_total_gpus': k * resource.get('num_gpus', 0),
                'conf_per_instance': {
                    'cpu_cores': resource.get('cpu_cores', 1),
                    'memory_gb': resource.get('memory_gb', 8),
                    'storage_gb': resource.get('storage_gb', 100),
                    'num_gpus': resource.get('num_gpus', 0),
                    'gpu_sm_count': resource.get('gpu_sm_count', 0),
                    'gpu_memory_gb': resource.get('gpu_memory_gb', 0),
                },
                'conf_resource_type': config.get('config_id', 0),
            }
            
            features_list.append(features)
        
        return features_list
    
    def generate(
        self,
        num_samples: int,
        k_range: tuple = (1, 16),
        output_format: str = 'yaml'
    ) -> Dict[str, Any]:
        """
        Generate configurations and extract features.
        
        Args:
            num_samples: Number of configurations
            k_range: Range for number of instances
            output_format: Output format ('yaml' or 'dict')
            
        Returns:
            Complete feature dictionary
        """
        # Generate LHS samples
        configurations = self.generate_lhs_samples(num_samples, k_range)
        
        # Extract features
        config_features = self.generate_config_features(configurations)
        
        # Build result
        result = {
            'configurations': [
                {
                    'config_id': c['config_id'],
                    'k': c['k'],
                    'resource': c['resource'],
                }
                for c in configurations
            ],
            'config_features': config_features,
            'metadata': {
                'num_samples': num_samples,
                'k_range': list(k_range),
                'catalog_size': self.n_resources,
                'sampling_method': 'latin_hypercube',
            },
            'catalog_stats': self.catalog_stats,
        }
        
        return result
    
    def save_to_yaml(
        self,
        result: Dict[str, Any],
        output_path: str
    ):
        """
        Save result to YAML file.
        
        Args:
            result: Result dictionary
            output_path: Output file path
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w', encoding='utf-8') as f:
            yaml.dump(result, f, default_flow_style=False, allow_unicode=True)


def load_resource_catalog(filepath: str) -> List[Dict[str, Any]]:
    """
    Load resource catalog from YAML file.
    
    Args:
        filepath: Path to catalog YAML file
        
    Returns:
        List of resource descriptions
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'resources' in data:
        return data['resources']
    else:
        raise ValueError("Invalid catalog format")


def generate_default_catalog() -> List[Dict[str, Any]]:
    """Generate a default cloud-like resource catalog."""
    return [
        # Small instances
        {'cpu_cores': 2, 'memory_gb': 4, 'storage_gb': 50, 'num_gpus': 0},
        {'cpu_cores': 4, 'memory_gb': 8, 'storage_gb': 100, 'num_gpus': 0},
        {'cpu_cores': 8, 'memory_gb': 16, 'storage_gb': 200, 'num_gpus': 0},
        
        # Medium instances
        {'cpu_cores': 16, 'memory_gb': 32, 'storage_gb': 500, 'num_gpus': 0},
        {'cpu_cores': 16, 'memory_gb': 64, 'storage_gb': 500, 'num_gpus': 1, 'gpu_memory_gb': 16},
        
        # Large instances
        {'cpu_cores': 32, 'memory_gb': 128, 'storage_gb': 1000, 'num_gpus': 0},
        {'cpu_cores': 32, 'memory_gb': 128, 'storage_gb': 1000, 'num_gpus': 4, 'gpu_memory_gb': 16},
        {'cpu_cores': 64, 'memory_gb': 256, 'storage_gb': 2000, 'num_gpus': 8, 'gpu_memory_gb': 32},
        
        # GPU-optimized
        {'cpu_cores': 8, 'memory_gb': 32, 'storage_gb': 200, 'num_gpus': 1, 'gpu_sm_count': 2560, 'gpu_memory_gb': 16},
        {'cpu_cores': 16, 'memory_gb': 64, 'storage_gb': 500, 'num_gpus': 2, 'gpu_sm_count': 5120, 'gpu_memory_gb': 32},
        {'cpu_cores': 32, 'memory_gb': 128, 'storage_gb': 1000, 'num_gpus': 4, 'gpu_sm_count': 10240, 'gpu_memory_gb': 40},
    ]


def generate_simple_catalog(
    cpu_cores: int,
    memory_gb: int,
    num_gpus: int = 0,
    gpu_memory_gb: int = 0,
    storage_gb: int = None
) -> List[Dict[str, Any]]:
    """
    Generate a simple catalog from command-line parameters.
    
    Creates 3 variations based on the specified resource.
    """
    if storage_gb is None:
        storage_gb = memory_gb * 10  # Default: 10x memory
    
    return [
        # Small variation (1/2 resources)
        {
            'cpu_cores': max(1, cpu_cores // 2),
            'memory_gb': max(1, memory_gb // 2),
            'storage_gb': max(1, storage_gb // 2),
            'num_gpus': max(0, num_gpus - 1),
            'gpu_memory_gb': gpu_memory_gb // 2 if gpu_memory_gb else 0,
            'gpu_sm_count': 0,
        },
        # Base configuration
        {
            'cpu_cores': cpu_cores,
            'memory_gb': memory_gb,
            'storage_gb': storage_gb,
            'num_gpus': num_gpus,
            'gpu_memory_gb': gpu_memory_gb,
            'gpu_sm_count': 0,
        },
        # Large variation (2x resources)
        {
            'cpu_cores': cpu_cores * 2,
            'memory_gb': memory_gb * 2,
            'storage_gb': storage_gb * 2,
            'num_gpus': num_gpus * 2,
            'gpu_memory_gb': gpu_memory_gb * 2 if gpu_memory_gb else 0,
            'gpu_sm_count': 0,
        },
    ]


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
        default='out/configs.yaml',
        help='Output YAML file path'
    )
    parser.add_argument(
        '--use-default-catalog',
        action='store_true',
        help='Use default cloud-like resource catalog'
    )
    
    args = parser.parse_args()
    
    # Load or generate resource catalog
    if args.resource_catalog:
        print(f"Loading resource catalog: {args.resource_catalog}")
        catalog = load_resource_catalog(args.resource_catalog)
    elif args.use_default_catalog:
        print("Using default cloud-like resource catalog")
        catalog = generate_default_catalog()
    else:
        print("Error: Please provide --resource-catalog or --use-default-catalog")
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
        print(f"  Config {i}: k={config['k']}, "
              f"CPU={r.get('cpu_cores', '?')} cores, "
              f"Mem={r.get('memory_gb', '?')}GB, "
              f"GPU={r.get('num_gpus', 0)}")


if __name__ == '__main__':
    from .config_generator_main import main
    main()
