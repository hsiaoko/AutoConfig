"""
Configuration Generator from System Capacity

Generates configurations constrained by system capacity.

Usage:
    autoconfig config --capacity data/system_capacity.yaml -n 10 -o out/configs.yaml
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from scipy.stats import qmc


class CapacityConstrainedConfigGenerator:
    """
    Generate configurations constrained by system capacity.
    
    System capacity defines the total available resources.
    Each configuration is a subset of the total capacity.
    """
    
    def __init__(self, capacity: Dict[str, Any]):
        """
        Initialize with system capacity.
        
        Args:
            capacity: System capacity dictionary
                {
                    'total_cpu_cores': 128,
                    'total_memory_gb': 512,
                    'total_storage_gb': 5000,
                    'total_gpus': 8,
                    'total_gpu_memory_gb': 256,
                    # Optional: machine configuration
                    'machine': {
                        'cpu_cores': 32,
                        'memory_gb': 128,
                        'storage_gb': 1000,
                        'num_gpus': 2,
                        'gpu_memory_gb': 32,
                    }
                }
        """
        self.capacity = capacity
        
        # Extract total resources
        self.total_cpu = capacity.get('total_cpu_cores', 64)
        self.total_memory = capacity.get('total_memory_gb', 256)
        self.total_storage = capacity.get('total_storage_gb', 2000)
        self.total_gpus = capacity.get('total_gpus', 0)
        self.total_gpu_memory = capacity.get('total_gpu_memory_gb', 0)
        
        # Extract machine configuration (if provided)
        self.machine = capacity.get('machine', None)
        
        # Calculate max machines
        if self.machine:
            self.max_machines_by_cpu = self.total_cpu // self.machine.get('cpu_cores', 1)
            self.max_machines_by_mem = self.total_memory // self.machine.get('memory_gb', 1)
            self.max_machines_by_gpu = self.total_gpus // self.machine.get('num_gpus', 1) if self.machine.get('num_gpus', 0) > 0 else self.max_machines_by_cpu
            self.max_machines = min(self.max_machines_by_cpu, self.max_machines_by_mem, self.max_machines_by_gpu)
        else:
            self.max_machines = self.total_cpu  # Fallback
        
        print(f"System capacity initialized:")
        print(f"  Total CPU: {self.total_cpu} cores")
        print(f"  Total Memory: {self.total_memory} GB")
        print(f"  Total Storage: {self.total_storage} GB")
        if self.total_gpus > 0:
            print(f"  Total GPUs: {self.total_gpus} ({self.total_gpu_memory} GB)")
        if self.machine:
            print(f"  Machine type: {self.machine.get('cpu_cores')}C/{self.machine.get('memory_gb')}G/{self.machine.get('num_gpus', 0)}GPU")
            print(f"  Max machines: {self.max_machines}")
    
    def generate_lhs_samples(
        self,
        num_samples: int,
        k_range: tuple = None
    ) -> List[Dict[str, Any]]:
        """
        Generate configurations using LHS, constrained by capacity.
        
        Args:
            num_samples: Number of configurations
            k_range: (min_k, max_k) - will be capped by capacity
            
        Returns:
            List of configuration dictionaries
        """
        # Determine k range based on capacity
        if k_range is None:
            k_min = 1
            k_max = self.max_machines
        else:
            k_min = k_range[0]
            k_max = min(k_range[1], self.max_machines)
        
        print(f"K range: [{k_min}, {k_max}] (capped by capacity: {self.max_machines})")
        
        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=1, seed=42)
        samples = sampler.random(n=num_samples)
        
        # Scale to k range
        k_values = np.floor(k_min + samples.flatten() * (k_max - k_min + 1)).astype(int)
        k_values = np.clip(k_values, k_min, k_max)
        
        # Generate configurations
        configurations = []
        for i, k in enumerate(k_values):
            config = {
                'config_id': i,
                'k': int(k),
                'resource': self._get_resource_config(k),
            }
            configurations.append(config)
        
        return configurations
    
    def _get_resource_config(self, k: int) -> Dict[str, Any]:
        """
        Get resource configuration for k machines.
        
        Args:
            k: Number of machines
            
        Returns:
            Resource dictionary
        """
        if self.machine:
            # Use machine configuration
            return {
                'cpu_cores': self.machine.get('cpu_cores', 1) * k,
                'memory_gb': self.machine.get('memory_gb', 8) * k,
                'storage_gb': self.machine.get('storage_gb', 100) * k,
                'num_gpus': self.machine.get('num_gpus', 0) * k,
                'gpu_memory_gb': self.machine.get('gpu_memory_gb', 0) * k,
                'gpu_sm_count': self.machine.get('gpu_sm_count', 0),
            }
        else:
            # Distribute capacity proportionally
            ratio = k / max(self.max_machines, 1)
            return {
                'cpu_cores': int(self.total_cpu * ratio),
                'memory_gb': int(self.total_memory * ratio),
                'storage_gb': int(self.total_storage * ratio),
                'num_gpus': int(self.total_gpus * ratio),
                'gpu_memory_gb': int(self.total_gpu_memory * ratio),
                'gpu_sm_count': 0,
            }
    
    def generate(
        self,
        num_samples: int,
        k_range: tuple = None
    ) -> Dict[str, Any]:
        """
        Generate configurations and extract features.
        
        Args:
            num_samples: Number of configurations
            k_range: (min_k, max_k) range
            
        Returns:
            Complete result dictionary
        """
        # Generate LHS samples
        configurations = self.generate_lhs_samples(num_samples, k_range)
        
        # Extract features
        config_features = self._extract_features(configurations)
        
        # Build result
        result = {
            'system_capacity': self.capacity,
            'configurations': configurations,
            'config_features': config_features,
            'metadata': {
                'num_samples': num_samples,
                'k_range': [
                    k_range[0] if k_range else 1,
                    k_range[1] if k_range else self.max_machines
                ],
                'max_machines': self.max_machines,
                'sampling_method': 'latin_hypercube_capacity_constrained',
            }
        }
        
        return result
    
    def _extract_features(self, configurations: List[Dict]) -> List[Dict]:
        """Extract features from configurations."""
        features_list = []
        
        for config in configurations:
            k = config['k']
            r = config['resource']
            
            features = {
                'conf_k_instances': k,
                'conf_total_cpu_cores': r.get('cpu_cores', 0),
                'conf_total_memory_gb': r.get('memory_gb', 0),
                'conf_total_storage_gb': r.get('storage_gb', 0),
                'conf_total_gpus': r.get('num_gpus', 0),
                'conf_per_instance': {
                    'cpu_cores': r.get('cpu_cores', 0) // max(k, 1),
                    'memory_gb': r.get('memory_gb', 0) // max(k, 1),
                    'storage_gb': r.get('storage_gb', 0) // max(k, 1),
                    'num_gpus': r.get('num_gpus', 0) // max(k, 1),
                    'gpu_memory_gb': r.get('gpu_memory_gb', 0) // max(k, 1),
                },
                'conf_memory_limit': r.get('memory_gb', 8) * 1024,  # MB
                'conf_num_threads': r.get('cpu_cores', 4),
                'conf_cache_size': r.get('memory_gb', 8) * 1024 // 8,
                'conf_batch_size': 1000,
                'conf_io_buffer_size': 64,
                'conf_num_workers': max(1, r.get('cpu_cores', 4) // 4),
                'conf_timeout': 300,
                'conf_enable_index': 1.0,
                'conf_index_type': 1.0,
                'conf_compression_enabled': 0.0,
            }
            
            features_list.append(features)
        
        return features_list
    
    def save_to_yaml(self, result: Dict[str, Any], output_path: str):
        """Save result to YAML file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w', encoding='utf-8') as f:
            yaml.dump(result, f, default_flow_style=False, allow_unicode=True)


def load_capacity(filepath: str) -> Dict[str, Any]:
    """Load system capacity from YAML file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Generate configurations constrained by system capacity'
    )
    parser.add_argument(
        '--capacity', '-c',
        type=str,
        required=True,
        help='System capacity YAML file'
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
        help='Minimum number of machines'
    )
    parser.add_argument(
        '--k-max',
        type=int,
        default=None,
        help='Maximum number of machines (default: capacity-limited)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='out/config_features.yaml',
        help='Output YAML file path'
    )
    
    args = parser.parse_args()
    
    # Load capacity
    print(f"Loading system capacity: {args.capacity}")
    capacity = load_capacity(args.capacity)
    
    # Create generator
    generator = CapacityConstrainedConfigGenerator(capacity)
    
    # Generate configurations
    k_range = (args.k_min, args.k_max) if args.k_max else None
    print(f"\nGenerating {args.num_samples} configurations using LHS...")
    
    result = generator.generate(args.num_samples, k_range)
    
    # Save to YAML
    generator.save_to_yaml(result, args.output)
    
    # Print summary
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


if __name__ == '__main__':
    main()
