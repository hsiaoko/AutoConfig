"""
Query Complexity Extractor

Extracts query complexity features from .cu/.cpp source files
(v_scan, e_scan, f_scan, atomic, sync — workload / analysis).

Complexity features:
- v_scan: Vertex scan operations (iterating over all vertices)
- e_scan: Edge scan operations (iterating over edges)
- f_scan: Frontier/neighbor scan operations
- atomic: Atomic operations
- sync: Synchronization barriers
"""

import re
from typing import Dict, Any
from pathlib import Path


class QueryComplexityExtractor:
    """
    Extract query complexity from CUDA/C++ source code.
    
    Analyzes code patterns to estimate computational complexity.
    """
    
    # Patterns for different operation types
    PATTERNS = {
        # Vertex scan: iterating over all vertices
        'v_scan': [
            r'for\s*\(\s*(?:int|size_t|auto)\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(?:numVertices|num_nodes|n_vertices|V)\s*',
            r'for\s*\(\s*(?:int|size_t|auto)\s+(\w+)\s*:\s*(?:vertices|all_nodes|nodes)\s*\)',
            r'for\s*\(\s*(?:int|size_t|auto)\s+(\w+)\s*=\s*(?:source|start)\s*;\s*\w+\s*<\s*(?:numVertices|n)',
            r'\bforAllVertices\b',
            r'\bparallel_for\s*\(\s*0\s*,\s*(?:numVertices|n)',
        ],
        
        # Edge scan: iterating over all edges
        'e_scan': [
            r'for\s*\(\s*(?:int|size_t|auto)\s+\w+\s*:\s*(?:edges|all_edges)\s*\)',
            r'for\s*\(\s*(?:int|size_t|auto)\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(?:numEdges|num_edges|m)\s*',
            r'\bforAllEdges\b',
            r'\bparallel_for\s*\(\s*0\s*,\s*(?:numEdges|m)',
        ],
        
        # Frontier scan: iterating over neighbors or worklist
        'f_scan': [
            r'for\s*\(\s*(?:int|size_t|auto)\s+\w+\s*:\s*(?:neighbors|adj|g\.adj)\s*\(',
            r'for\s*\(\s*(?:int|size_t|auto)\s+\w+\s*:\s*worklist\b',
            r'for\s*\(\s*(?:int|size_t|auto)\s+\w+\s*:\s*frontier\b',
            r'for\s*\(\s*(?:int|size_t|auto)\s+(\w+)\s*=\s*g\.offset\[\s*\w+\s*\]\s*;\s*\w+\s*<\s*g\.offset\[\s*\w+\s*\+\s*1\]',
            r'for\s*\(\s*(?:int|size_t|auto)\s+(\w+)\s*=\s*rowOffset\[\s*\w+\s*\]\s*;',
            r'\bforNeighbors\b',
            r'\bforFrontier\b',
            r'\bforActive\b',
        ],
        
        # Atomic operations
        'atomic': [
            r'\batomicAdd\b',
            r'\batomicSub\b',
            r'\batomicMax\b',
            r'\batomicMin\b',
            r'\batomicCAS\b',
            r'\batomicExch\b',
            r'\bfetch_add\b',
            r'\bfetch_sub\b',
            r'\b__syncthreads\b',  # CUDA sync often paired with atomics
        ],
        
        # Synchronization barriers
        'sync': [
            r'\bbarrier\b',
            r'\bsync\b',
            r'\b__syncthreads\b',
            r'\bMPI_Barrier\b',
            r'\bMPI_Allreduce\b',
            r'\bMPI_Reduce\b',
            r'\bomp_barrier\b',
            r'\b#pragma\s+omp\s+barrier\b',
        ],
    }
    
    def __init__(self):
        self.patterns = {
            name: [re.compile(p, re.IGNORECASE) for p in patterns]
            for name, patterns in self.PATTERNS.items()
        }
    
    def extract(self, source_code: str) -> Dict[str, int]:
        """
        Extract complexity features from source code.
        
        Args:
            source_code: CUDA/C++ source code string
            
        Returns:
            Dictionary with complexity counts
        """
        complexity = {
            'v_scan': 0,
            'e_scan': 0,
            'f_scan': 0,
            'atomic': 0,
            'sync': 0,
        }
        
        for feature_name, compiled_patterns in self.patterns.items():
            count = 0
            for pattern in compiled_patterns:
                matches = pattern.findall(source_code)
                count += len(matches)
            complexity[feature_name] = count
        
        # Ensure at least 1 for non-zero complexity
        total = sum(complexity.values())
        if total == 0:
            # Default: assume at least some vertex and edge scanning
            complexity['v_scan'] = 1
            complexity['e_scan'] = 1
        
        return complexity
    
    def extract_from_file(self, filepath: str) -> Dict[str, int]:
        """
        Extract complexity from a source file.
        
        Args:
            filepath: Path to .cu or .cpp file
            
        Returns:
            Complexity dictionary
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        return self.extract(source_code)
    
    def get_base_cost(self, complexity: Dict[str, int]) -> float:
        """
        Estimate base computational cost from complexity.
        
        Args:
            complexity: Complexity dictionary
            
        Returns:
            Base cost estimate
        """
        # Weight different operations
        weights = {
            'v_scan': 10.0,
            'e_scan': 50.0,
            'f_scan': 100.0,
            'atomic': 80.0,
            'sync': 20.0,
        }
        
        base_cost = sum(complexity[k] * weights[k] for k in complexity)
        return base_cost


def extract_query_complexity(filepath: str) -> Dict[str, Any]:
    """
    Convenience function to extract query complexity from file.
    
    Args:
        filepath: Path to .cu or .cpp file
        
    Returns:
        Dictionary with complexity and base_cost
    """
    extractor = QueryComplexityExtractor()
    complexity = extractor.extract_from_file(filepath)
    base_cost = extractor.get_base_cost(complexity)
    
    return {
        'complexity': complexity,
        'base_cost': base_cost,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract query complexity from CUDA/C++ source file'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input .cu or .cpp file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output YAML file (optional)'
    )
    
    args = parser.parse_args()
    
    result = extract_query_complexity(args.input)
    
    print(f"Query complexity extracted from {args.input}:")
    for k, v in result['complexity'].items():
        print(f"  {k}: {v}")
    print(f"  base_cost: {result['base_cost']:.2f}")
    
    if args.output:
        import yaml
        with open(args.output, 'w') as f:
            yaml.dump(result, f, default_flow_style=None)
        print(f"Saved to {args.output}")
