"""
Static Feature Extractor
Extracts structural features from query source code.

Corresponds to Table 1 (Static features 1-8):
1. Loop count
2. Maximum loop depth
3. Branch count
4. Variable count
5. Recursion count
6. Atomic operation count
7. Synchronization count
8. Explicit parallel flag
"""

import re
import ast
from typing import Dict, Any, List, Tuple
import numpy as np


class StaticFeatureExtractor:
    """
    Extracts static structural features from query source code.
    
    Analyzes code structure independent of input graph.
    """
    
    def __init__(self):
        self.feature_names = [
            'static_loop_count',
            'static_max_loop_depth',
            'static_branch_count',
            'static_variable_count',
            'static_recursion_count',
            'static_atomic_op_count',
            'static_sync_count',
            'static_explicit_parallel_flag',
        ]
        
        # Patterns for different languages (including pseudo-code and C++/CUDA)
        self.patterns = {
            # Loop patterns
            'loop_for': r'\bfor\s*\(',
            'loop_for_range': r'\bfor\s*\(\s*\w+\s*[=:]\s*\d+\s*;\s*\w+\s*<',
            'loop_for_each': r'\bfor\s*\(\s*\w+\s+\w+\s*:\s*\w+\s*\)',
            'loop_while': r'\bwhile\s*\(',
            'loop_foreach': r'\bfor\s*\(\s*\w+\s+in\s+\w+\.\w+\s*\)',
            'loop_prism': r'\bfor\s*\(\s*\w+\s+in\s+\w+\.\w+\s*\)',
            'loop_cuda': r'<<<[^>]*>>>',  # CUDA kernel launch in loop
            
            # Branch patterns
            'branch_if': r'\bif\s*\(',
            'branch_if_cpp': r'\bif\s*\(\s*[^)]+\s*\)\s*\{',
            'branch_switch': r'\bswitch\s*\(',
            'branch_elif': r'\belif\s*\(',
            'branch_else_if': r'\belse\s+if\s*\(',
            
            # Variable patterns
            'var_decl': r'(?:int|float|double|char|bool|auto|var|let|uint32_t|int32_t|size_t|void)\s+\*?\s*\w+',
            'var_decl_cpp': r'(?:const|static|volatile)\s+(?:\w+\s+)+\w+',
            'var_cuda_buffer': r'\b(?:DeviceOwnedBuffer|Buffer|deviceMalloc|cudaMalloc)\b',
            
            # Recursion (detected via function calls with same name)
            'func_def': r'(?:def|function)\s+(\w+)\s*[:(]',
            'func_def_cpp': r'^\s*(?:static\s+)?(?:__forceinline__\s+)?(?:__device__\s+)?\w+\s+\w+\s*\([^)]*\)\s*\{',
            'func_def_cpp_method': r'^\s*\w+\s+\w+::\w+\s*\([^)]*\)',
            'func_kernel': r'\b__global__\s+\w+\s+\w+\s*\(',
            
            # Atomic operations
            'atomic_add': r'\b(atomicAdd|atomic_add|fetch_add|atomicInc)\b',
            'atomic_cas': r'\b(atomicCAS|atomic_compare_exchange|compare_and_swap)\b',
            'atomic_store': r'\b(atomicStore|atomic_store|fetch_store|atomicExch)\b',
            'atomic_cuda': r'\batomic(Add|Sub|Max|Min|And|Or|Xor|CAS|Exch)\b',
            
            # Synchronization
            'sync_barrier': r'\b(barrier|__syncthreads|barrier_sync|cudaDeviceSynchronize)\b',
            'sync_lock': r'\b(lock|mutex\.lock|pthread_mutex_lock|std::lock_guard)\b',
            'sync_wait': r'\b(wait|join|await|cudaStreamSynchronize)\b',
            'sync_cuda': r'\b(cudaDeviceSynchronize|cudaStreamSynchronize|cudaEventSynchronize)\b',
            
            # Parallel constructs
            'parallel_pragma': r'#pragma\s+(omp\s+parallel|parallel)',
            'parallel_for': r'#pragma\s+omp\s+parallel\s+for',
            'parallel_launch': r'\b(parallel_for|parallel_invoke|spawn)\b',
            'cuda_kernel_launch': r'<<<\s*\w+\s*,\s*\w+\s*>>>',
            'cuda_kernel_call': r'\w+Kernel\s*<<<',
            
            # CUDA specific
            'cuda_memory': r'\b(cudaMalloc|cudaFree|cudaMemcpy|cudaMemset)\b',
            'cuda_thread': r'\b(threadIdx|blockIdx|blockDim|gridDim)\b',
            'cuda_shared': r'\b__shared__\s+\w+\s+\w+',
        }
    
    def extract(self, source_code: str) -> np.ndarray:
        """
        Extract static features from source code.
        
        Args:
            source_code: Source code string
            
        Returns:
            numpy array of static features
        """
        features = []
        
        # 1. Loop count
        loop_count = self._count_loops(source_code)
        features.append(float(loop_count))
        
        # 2. Maximum loop depth
        max_loop_depth = self._compute_max_loop_depth(source_code)
        features.append(float(max_loop_depth))
        
        # 3. Branch count
        branch_count = self._count_branches(source_code)
        features.append(float(branch_count))
        
        # 4. Variable count
        var_count = self._count_variables(source_code)
        features.append(float(var_count))
        
        # 5. Recursion count
        recursion_count = self._count_recursive_functions(source_code)
        features.append(float(recursion_count))
        
        # 6. Atomic operation count
        atomic_count = self._count_atomic_ops(source_code)
        features.append(float(atomic_count))
        
        # 7. Synchronization count
        sync_count = self._count_synchronization(source_code)
        features.append(float(sync_count))
        
        # 8. Explicit parallel flag
        parallel_flag = 1.0 if self._has_explicit_parallel(source_code) else 0.0
        features.append(parallel_flag)
        
        return np.array(features, dtype=np.float64)
    
    def _count_loops(self, code: str) -> int:
        """Count loop constructs."""
        count = 0
        for pattern in ['loop_for', 'loop_for_range', 'loop_for_each', 'loop_while', 
                        'loop_foreach', 'loop_prism']:
            count += len(re.findall(self.patterns[pattern], code, re.IGNORECASE | re.MULTILINE))
        return count
    
    def _compute_max_loop_depth(self, code: str) -> int:
        """Compute maximum loop nesting depth using braces and indentation."""
        lines = code.split('\n')
        max_depth = 0
        current_depth = 0
        in_loop = []
        
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            # Check if line starts a loop
            is_loop_start = any(
                re.search(self.patterns[p], stripped)
                for p in ['loop_for', 'loop_for_range', 'loop_for_each', 'loop_while',
                          'loop_foreach', 'loop_prism']
            )
            
            if is_loop_start:
                in_loop.append(indent)
                current_depth = len(in_loop)
                max_depth = max(max_depth, current_depth)
            
            # Track depth by braces
            open_braces = stripped.count('{')
            close_braces = stripped.count('}')
            
            if close_braces > 0 and in_loop:
                for _ in range(close_braces):
                    if in_loop and indent <= in_loop[-1]:
                        in_loop.pop()
                        current_depth = len(in_loop)
            
            if open_braces > 0 and is_loop_start:
                current_depth = len(in_loop)
        
        return max_depth
    
    def _count_branches(self, code: str) -> int:
        """Count conditional branches."""
        count = 0
        for pattern in ['branch_if', 'branch_if_cpp', 'branch_switch', 'branch_elif', 'branch_else_if']:
            count += len(re.findall(self.patterns[pattern], code, re.IGNORECASE | re.MULTILINE))
        return count
    
    def _count_variables(self, code: str) -> int:
        """Count variable definitions."""
        # Type-based declarations (including C++ types)
        count = len(re.findall(self.patterns['var_decl'], code))
        
        # C++ style declarations
        count += len(re.findall(self.patterns['var_decl_cpp'], code))
        
        # CUDA buffer declarations
        count += len(re.findall(self.patterns['var_cuda_buffer'], code))
        
        # Python-style assignments (simple heuristic)
        python_assigns = len(re.findall(self.patterns.get('var_assign', r'^\s*\w+\s*=\s*'), code, re.MULTILINE))
        
        # Pseudo-code style: variable declarations in function params
        param_vars = len(re.findall(r'\b(?:Fragment|Context|Match|Pattern|Graph|Buffer|Device)\s+\w+', code))
        
        return max(count, python_assigns) + param_vars
    
    def _count_recursive_functions(self, code: str) -> int:
        """Count recursive function definitions."""
        # Find function definitions
        func_defs = re.findall(self.patterns['func_def'], code)
        
        recursion_count = 0
        for func_name in func_defs:
            # Check if function calls itself
            # Simple heuristic: function name appears in its own body
            pattern = rf'\b{func_name}\s*\('
            matches = re.findall(pattern, code)
            if len(matches) > 1:  # More than just the definition
                recursion_count += 1
        
        return recursion_count
    
    def _count_atomic_ops(self, code: str) -> int:
        """Count atomic operations."""
        count = 0
        for pattern in ['atomic_add', 'atomic_cas', 'atomic_store', 'atomic_cuda']:
            count += len(re.findall(self.patterns[pattern], code, re.IGNORECASE))
        return count
    
    def _count_synchronization(self, code: str) -> int:
        """Count synchronization primitives."""
        count = 0
        for pattern in ['sync_barrier', 'sync_lock', 'sync_wait', 'sync_cuda']:
            count += len(re.findall(self.patterns[pattern], code, re.IGNORECASE))
        return count
    
    def _has_explicit_parallel(self, code: str) -> bool:
        """Check for explicit parallel constructs."""
        # Check for CUDA kernel launches
        if re.search(self.patterns['cuda_kernel_launch'], code):
            return True
        if re.search(self.patterns['cuda_kernel_call'], code):
            return True
        
        # Check for OpenMP and other parallel constructs
        for pattern in ['parallel_pragma', 'parallel_for', 'parallel_launch']:
            if re.search(self.patterns[pattern], code, re.IGNORECASE):
                return True
        
        # Check for CUDA thread indexing
        if re.search(self.patterns['cuda_thread'], code):
            return True
        
        return False
    
    def extract_from_file(self, filepath: str) -> np.ndarray:
        """
        Extract features from a source file.
        
        Args:
            filepath: Path to source file
            
        Returns:
            numpy array of static features
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        return self.extract(source_code)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()


class ASTBasedStaticExtractor:
    """
    AST-based static feature extractor for Python code.
    More accurate than regex-based approach.
    """
    
    def __init__(self):
        self.feature_names = StaticFeatureExtractor().feature_names
    
    def extract(self, source_code: str) -> np.ndarray:
        """Extract features using AST analysis."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # Fallback to regex-based extractor
            return StaticFeatureExtractor().extract(source_code)
        
        features = [
            float(self._count_loops(tree)),
            float(self._compute_max_loop_depth(tree)),
            float(self._count_branches(tree)),
            float(self._count_variables(tree)),
            float(self._count_recursion(tree, source_code)),
            float(self._count_atomic_ops(source_code)),
            float(self._count_sync_ops(source_code)),
            float(1.0 if self._has_parallel(tree) else 0.0),
        ]
        
        return np.array(features, dtype=np.float64)
    
    def _count_loops(self, tree: ast.AST) -> int:
        """Count loop nodes in AST."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                count += 1
        return count
    
    def _compute_max_loop_depth(self, tree: ast.AST) -> int:
        """Compute maximum loop nesting depth."""
        def get_depth(node: ast.AST, current_depth: int) -> int:
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            return max_depth
        
        return get_depth(tree, 0)
    
    def _count_branches(self, tree: ast.AST) -> int:
        """Count branch nodes in AST."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.Match)):
                count += 1
        return count
    
    def _count_variables(self, tree: ast.AST) -> int:
        """Count variable definitions in AST."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                count += len(node.targets) if hasattr(node, 'targets') else 1
        return count
    
    def _count_recursion(self, tree: ast.AST, source_code: str) -> int:
        """Count recursive functions."""
        func_names = set()
        
        # Collect function names
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_names.add(node.name)
        
        # Check for self-calls
        recursion_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in func_names:
                    recursion_count += 1
        
        return max(0, recursion_count - len(func_names))  # Subtract definitions
    
    def _count_atomic_ops(self, source_code: str) -> int:
        """Count atomic operations (regex fallback)."""
        patterns = [
            r'\b(atomicAdd|atomic_add|fetch_add)\b',
            r'\b(atomicCAS|atomic_compare_exchange)\b',
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, source_code, re.IGNORECASE))
        return count
    
    def _count_sync_ops(self, source_code: str) -> int:
        """Count synchronization operations (regex fallback)."""
        patterns = [
            r'\b(barrier|__syncthreads)\b',
            r'\b(lock|mutex\.lock)\b',
            r'\b(wait|join|await)\b',
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, source_code, re.IGNORECASE))
        return count
    
    def _has_parallel(self, tree: ast.AST) -> bool:
        """Check for parallel constructs in AST."""
        for node in ast.walk(tree):
            # Check for multiprocessing/async calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['start', 'submit', 'map_async']:
                        return True
        return False
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()
