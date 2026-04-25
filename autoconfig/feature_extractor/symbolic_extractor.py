"""
Symbolic Feature Extractor
Extracts symbolic workload templates from query code.

Corresponds to Table 1 (Symbolic features 1-6):
1. Vertex scanning (VScan)
2. Edge scanning (EScan)
3. Frontier iteration (FScan)
4. Recursive expansion (RExp)
5. Atomic state update (Atom)
6. Cross-partition communication (Comm)
"""

import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Paper Table: symbolic workload templates (formulas are instantiated in the merge stage).
SYMBOLIC_FAMILY_IDS = ("vscan", "escan", "fscan", "rexp", "atom", "comm")


class SymbolicFeatureExtractor:
    """
    Extracts symbolic workload templates from query source code.
    
    Identifies performance-critical patterns and creates partial functions
    that are instantiated with graph/partition statistics.
    """
    
    def __init__(self):
        self.feature_names = [
            'sym_vscan_coeff',      # Vertex scanning coefficient
            'sym_vscan_requires',   # Requires |V| flag
            'sym_escan_coeff',      # Edge scanning coefficient
            'sym_escan_requires',   # Requires |E| flag
            'sym_fscan_coeff',      # Frontier iteration coefficient
            'sym_fscan_requires',   # Requires diameter flag
            'sym_rexp_coeff',       # Recursive expansion coefficient
            'sym_rexp_requires',    # Requires diameter flag
            'sym_atom_coeff',       # Atomic update coefficient
            'sym_atom_requires',    # Requires skew flag
            'sym_comm_coeff',       # Cross-partition communication coefficient
            'sym_comm_requires',    # Requires boundary-degree flag
        ]
        
        # Code patterns for symbolic feature detection (including pseudo-code and C++/CUDA)
        self.patterns = {
            # Vertex scanning patterns
            'vertex_scan': [
                r'\bfor\s*\(\s*\w+\s+in\s+\w*\.?vertices\b',
                r'\bfor\s*\(\s*\w+\s*:\s*\w*\.?vertices\b',
                r'\bfor\s*\([^;]+;\s*[^;]+<\s*v_?count\b',
                r'\.Vertices\s*\(\s*\)',
                r'\bgraph\s*\.?\s*vertices\s*\(\s*\)',
                r'\bfor_each_vertex\s*\(',
                r'\bvertex_range\s*\(',
                r'\bF\s*\.?\s*Vertices\s*\(\s*\)',
                r'\bg\.n_vertices\b',
                r'\bgraph\.n_vertices\b',
            ],
            
            # Edge scanning patterns
            'edge_scan': [
                r'\bfor\s*\(\s*\w+\s+in\s+\w*\.?edges\b',
                r'\bfor\s*\(\s*\w+\s*:\s*\w*\.?edges\b',
                r'\bfor\s*\([^;]+;\s*[^;]+<\s*e_?count\b',
                r'\.Edges\s*\(\s*\)',
                r'\bgraph\s*\.?\s*edges\s*\(\s*\)',
                r'\bfor_each_edge\s*\(',
                r'\bneighbor\s*\(\s*\w+\s*\)',
                r'\badjacent\s*\(\s*\w+\s*\)',
                r'\bout_edges\s*\(',
                r'\bin_edges\s*\(',
                r'\bG\s*\.?\s*neighbors\s*\(',
                r'\bG\s*\.?\s*out_edges\s*\(',
                r'\bG\s*\.?\s*in_edges\s*\(',
                r'\bg\.e_src\b',
                r'\bg\.e_dst\b',
                r'\bg\.n_edges\b',
                r'\bgraph\.n_edges\b',
            ],
            
            # Frontier iteration patterns
            'frontier_scan': [
                r'\bwhile\s+!\s*\w*\.?\s*empty\s*\(\s*\)',
                r'\bwhile\s*\(\s*!\s*\w*\.?\s*empty\s*\(\s*\)\s*\)',
                r'\bwhile\s+!\s*\w*\.?\s*Empty\s*\(\s*\)',
                r'\bworklist\s*\.\s*(?:pop|front|dequeue|empty)\s*\(',
                r'\bfrontier\s*\.\s*(?:next|advance|iterate)\s*\(',
                r'\bactive_set\s*\.\s*(?:update|process)\s*\(',
                r'\bfor\s+.*\s+in\s+1\s*\.\.\s*\w*',
                r'\biteration\s*<\s*\w*\.?\s*diameter\b',
                r'\bcurr_count\s*>\s*0\b',
                r'\bnext_count\s*>\s*0\b',
            ],
            
            # Recursive expansion patterns
            'recursive_exp': [
                r'\bExpand\s*\([^)]*level\s*\+\s*1',
                r'\brecursive\s*\(\s*\w+\s*,\s*\w+\s*\+\s*1',
                r'\bdfs\s*\([^)]*depth\s*\+\s*1',
                r'\bbfs\s*\([^)]*level\s*\+\s*1',
                r'\bsearch\s*\([^)]*step\s*\+\s*1',
                r'\bGARExpand\b',
                r'\bExpandEmbeddings\b',
            ],
            
            # Atomic update patterns
            'atomic_update': [
                r'\batomicAdd\s*\(',
                r'\batomic_add\s*\(',
                r'\batomicCAS\s*\(',
                r'\bcompare_and_swap\s*\(',
                r'\bfetch_add\s*\(',
                r'\batomic\s*<\s*\w+\s*>\s*::\s*add\s*\(',
                r'\batomic(Add|Sub|Max|Min|Inc)\b',
            ],
            
            # Cross-partition communication patterns
            'cross_partition': [
                r'\bSendTo\s*\(',
                r'\bsend_message\s*\(',
                r'\bcommunicate\s*\(\s*\w+\s*,\s*partition',
                r'\bmirror\s*\.\s*(?:owner|partition)\s*\(',
                r'\bboundary\s*\.\s*(?:send|communicate)\s*\(',
                r'\bIsMirror\s*\(',
                r'\bOwner\s*\(',
                r'\bremote\s*\.\s*(?:send|push)\s*\(',
                r'\bctx\s*\.?\s*SendTo\b',
                r'\bcontext\s*\.?\s*Owner\b',
            ],
            
            # CUDA specific patterns
            'cuda_vertex_processing': [
                r'\bfor\s*\(\s*(?:unsigned\s+)?(?:int|uint)\s+\w+\s*=\s*_tid\b',
                r'\bthreadIdx\.x\b',
                r'\bblockIdx\.x\b',
                r'\bblockDim\.x\b',
                r'\bgridDim\.x\b',
                r'\bstep\s*=\s*blockDim\.x\s*\*\s*gridDim\.x\b',
                r'\bfor\s*\([^;]+;\s*\w+\s*<\s*n_vertices\b',
            ],
            
            'cuda_edge_processing': [
                r'\bfor\s*\([^;]+;\s*\w+\s*<\s*n_edges\b',
                r'\bge\s*<\s*params\.n_edges\b',
                r'\bfor\s*\([^;]+;\s*\w+\s*<\s*params\.n_edges\b',
            ],
        }

        # Heuristic patterns: map code structure to symbolic families without requiring
        # explicit G.vertices() / VScan keywords. Goal: workload ≈ computational cost.
        # - Vertex-scoped loops / vertex streaming APIs → VScan
        # - Edge lists, neighbor loops, edge streaming APIs → EScan
        # - Frontiers, active sets, worklists, BFS-style rounds → FScan (diameter-scaling)
        # - Recursion / explicit depth-level expansion → RExp (degree^diameter-style)
        # - Atomics / write_add → Atom (merged with |E|·skew for contention proxy)
        self.heuristic_patterns = {
            'vertex_loops': [
                r'\bstream_vertices\s*<',
                r'\bfor_each_vertex\s*\(',
                r'for\s*\(\s*VertexID\s+\w+\s*=\s*[^;]+;\s*\w+\s*<\s*(?:params\.)?n_vertices\w*',
                r'for\s*\([^)]*<\s*(?:params\.)?n_vertices_g\b',
                r'for\s*\([^)]*<\s*(?:g|graph)\.get_num_vertices\s*\(',
                r'\bv_idx\s*<\s*(?:params\.)?n_vertices\w*',
                r'for\s*\(\s*[^;]+;\s*[^;]+<\s*[^;)]*max_vid\b',
                r'for\s*\(\s*int\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*graph\.vertices\b',
            ],
            'edge_loops': [
                r'\bstream_edges\s*<',
                r'for\s*\([^)]*nbr_idx[^)]*<\s*(?:in_|out_)?degree',
                r'for\s*\([^)]*<\s*(?:params\.)?n_edges_g\b',
                r'for\s*\([^)]*e_idx[^)]*<\s*(?:params\.)?n_edges\w*',
                r'\bin_edges_g\s*\[',
                r'\bout_edges_g\s*\[',
                r'Edge\s*&\s*e\b',
                r'\[\s*&\s*\]\s*\(\s*Edge\s*&',
                r'for\s*\([^)]*<\s*(?:in_offset|out_offset)_g\s*\[',
            ],
            'frontier_diameter': [
                r'\bwhile\s*\(\s*!\s*\w+\s*\.\s*(?:empty|Empty)\s*\(',
                r'\b(?:worklist|frontier|next_frontier|curr_frontier|active_set)\b',
                r'\bqueue\s*<',  # std::queue, work queue
                r'\b(?:in_active|out_active)_vertices\b',
                r'\b(?:BFS|bfs|propagat(?:e|ion))\b',
                r'while\s*\([^)]*(?:frontier|active_?set|n_active|work_size)\b[^)]*[><=!]',
            ],
            'recursion_diameter': [
                r'\bvoid\s+(?:dfs|bfs|dfs_visit|bfs_visit)\b',
                r'\b(?:int|void|bool|auto)\s+(?:dfs|bfs)\b',
                r'\bdfs\s*\(|\bDFS\s*\(',
                r'\bdepth\s*\+\s*1',
                r'\blevel\s*\+\s*1',
                r'Expand\s*\([^)]*level\s*\+\s*1',  # align with static recursive_exp
                r'\brecursive\b',
            ],
            'atomic_skew': [
                r'\bwrite_add\s*\(',
                r'__atomic_|__sync_lock|#pragma\s+omp\s+atomic',
            ],
        }

    def _heuristic_any(self, code: str, pattern_keys: List[str]) -> bool:
        for key in pattern_keys:
            for pat in self.heuristic_patterns[key]:
                if re.search(pat, code, re.IGNORECASE | re.MULTILINE):
                    return True
        return False

    def extract_templates(
        self,
        source_code: str
    ) -> np.ndarray:
        """
        Extract symbolic feature templates without graph instantiation.
        
        Deprecated for YAML output: use ``extract_symbolic_expressions`` so that
        symbolic features are stored as template formulas, not placeholder floats.
        This method remains for backward compatibility with numeric pipelines.
        """
        expr = self.extract_symbolic_expressions(source_code)
        families = expr["families"]
        order = SYMBOLIC_FAMILY_IDS
        features = []
        for fid in order:
            fam = families[fid]
            features.append(1.0 if fam["detected"] else 0.0)
            features.append(1.0 if fam["detected"] else 0.0)
        return np.array(features, dtype=np.float64)

    def extract_symbolic_expressions(
        self,
        source_code: str,
    ) -> Dict[str, Any]:
        """
        Stage-wise symbolic templates: partial functions expressed as formulas.

        Static analysis only identifies which templates apply; concrete values
        are produced when graph/partition statistics are supplied (merge step).

        Returns:
            Dict with ``families`` keyed by vscan, escan, fscan, rexp, atom, comm.
        """
        families: Dict[str, Dict[str, Any]] = {
            "vscan": {
                "template": "VScan(V, n)",
                "formula": "|V| / n",
                "latex": r"\mathrm{VScan}(V,n): |V| \text{ or } |V|/n",
                "detected": self._detect_vertex_scan(source_code),
                "requires_graph": ["num_vertices"],
                "requires_partition": [],
            },
            "escan": {
                "template": "EScan(E, n)",
                "formula": "|E| / n",
                "latex": r"\mathrm{EScan}(E,n): |E| \text{ or } |E|/n",
                "detected": self._detect_edge_scan(source_code),
                "requires_graph": ["num_edges"],
                "requires_partition": [],
            },
            "fscan": {
                "template": "FScan(G, n)",
                "formula": "sum_{t=1}^{D} |E_t|",
                "latex": r"\mathrm{FScan}(G,n): \sum_{t=1}^{\mathbb{D}} |E_t|",
                "detected": self._detect_frontier_scan(source_code),
                "requires_graph": ["num_edges", "diameter"],
                "requires_partition": [],
            },
            "rexp": {
                "template": "RExp(G)",
                "formula": "prod_{i=1}^{D} E[deg(v_i)]",
                "latex": r"\mathrm{RExp}(G): \prod_{i=1}^{\mathbb{D}} \mathbb{E}[\deg(v_i)]",
                "detected": self._detect_recursive_exp(source_code),
                "requires_graph": ["avg_degree", "diameter"],
                "requires_partition": [],
            },
            "atom": {
                "template": "Atom(G)",
                "formula": "|E| * skew(G)",
                "latex": r"\mathrm{Atom}(G): |E| \cdot \mathrm{skew}(G)",
                "detected": self._detect_atomic_update(source_code),
                "requires_graph": ["num_edges", "skew"],
                "requires_partition": [],
            },
            "comm": {
                "template": "Comm(G, F)",
                "formula": "sum_{v in V_d} deg_d(v)",
                "latex": r"\mathrm{Comm}(G,\mathcal{F}): \sum_{v \in V_\partial} \deg_\partial(v)",
                "detected": self._detect_cross_partition(source_code),
                "requires_graph": [],
                "requires_partition": ["boundary_degree_sum"],
            },
        }
        return {
            "format_version": 2,
            "families": families,
        }

    def extract(
        self,
        source_code: str,
        graph_stats: Optional[Dict[str, Any]] = None,
        partition_stats: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract symbolic features from source code.
        
        Args:
            source_code: Source code string
            graph_stats: Optional graph statistics for instantiation
            partition_stats: Optional partition statistics for instantiation
            
        Returns:
            numpy array of symbolic features (coefficients and flags)
        """
        features = []
        
        # 1. Vertex scanning
        vscan_coeff, vscan_req = self._extract_vertex_scan(source_code, graph_stats)
        features.append(vscan_coeff)
        features.append(1.0 if vscan_req else 0.0)
        
        # 2. Edge scanning
        escan_coeff, escan_req = self._extract_edge_scan(source_code, graph_stats)
        features.append(escan_coeff)
        features.append(1.0 if escan_req else 0.0)
        
        # 3. Frontier iteration
        fscan_coeff, fscan_req = self._extract_frontier_scan(source_code, graph_stats)
        features.append(fscan_coeff)
        features.append(1.0 if fscan_req else 0.0)
        
        # 4. Recursive expansion
        rexp_coeff, rexp_req = self._extract_recursive_exp(source_code, graph_stats)
        features.append(rexp_coeff)
        features.append(1.0 if rexp_req else 0.0)
        
        # 5. Atomic update
        atom_coeff, atom_req = self._extract_atomic_update(source_code, graph_stats)
        features.append(atom_coeff)
        features.append(1.0 if atom_req else 0.0)
        
        # 6. Cross-partition communication
        comm_coeff, comm_req = self._extract_cross_partition(
            source_code, graph_stats, partition_stats
        )
        features.append(comm_coeff)
        features.append(1.0 if comm_req else 0.0)
        
        return np.array(features, dtype=np.float64)

    # Detection methods (without graph instantiation)
    
    def _detect_vertex_scan(self, code: str) -> bool:
        """Detect vertex scanning: explicit API names or loops bounded by |V|."""
        for pattern in self.patterns['vertex_scan'] + self.patterns.get(
            'cuda_vertex_processing', []
        ):
            if re.search(pattern, code, re.IGNORECASE):
                return True
        if self._heuristic_any(code, ['vertex_loops']):
            return True
        return False
    
    def _detect_edge_scan(self, code: str) -> bool:
        """Detect edge / neighbor iteration: explicit APIs or loops over |E| / adjacency."""
        for pattern in self.patterns['edge_scan'] + self.patterns.get(
            'cuda_edge_processing', []
        ):
            if re.search(pattern, code, re.IGNORECASE):
                return True
        if self._heuristic_any(code, ['edge_loops']):
            return True
        return False
    
    def _detect_frontier_scan(self, code: str) -> bool:
        """Detect worklist / frontier / active-set iteration (diameter-scaling work)."""
        for pattern in self.patterns['frontier_scan']:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        if self._heuristic_any(code, ['frontier_diameter']):
            return True
        return False
    
    def _detect_recursive_exp(self, code: str) -> bool:
        """Detect recursion or depth-incremental expansion (typical of degree^d cost)."""
        for pattern in self.patterns['recursive_exp']:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        if self._heuristic_any(code, ['recursion_diameter']):
            return True
        return False
    
    def _detect_atomic_update(self, code: str) -> bool:
        """Detect atomics and lock-free updates (paired with |E|·skew for skewed contention)."""
        for pattern in self.patterns['atomic_update']:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        if self._heuristic_any(code, ['atomic_skew']):
            return True
        return False
    
    def _detect_cross_partition(self, code: str) -> bool:
        """Detect cross-partition communication patterns."""
        for pattern in self.patterns['cross_partition']:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False

    def _extract_vertex_scan(
        self,
        code: str,
        graph_stats: Optional[Dict[str, Any]]
    ) -> Tuple[float, bool]:
        """
        Extract vertex scanning feature.
        
        Returns:
            (coefficient, requires_vertex_count)
        """
        # Count vertex scan patterns
        count = 0
        for pattern in self.patterns['vertex_scan']:
            count += len(re.findall(pattern, code, re.IGNORECASE))
        
        if count == 0:
            return 0.0, False
        
        # If graph stats available, compute actual cost
        if graph_stats and 'num_vertices' in graph_stats:
            # VScan(V, n) = |V| or |V|/n
            n = graph_stats.get('vertex_scan_factor', 1)
            coeff = graph_stats['num_vertices'] / n
            return float(coeff), True
        
        # Otherwise return pattern count as coefficient
        return float(count), True
    
    def _extract_edge_scan(
        self,
        code: str,
        graph_stats: Optional[Dict[str, Any]]
    ) -> Tuple[float, bool]:
        """
        Extract edge scanning feature.
        
        Returns:
            (coefficient, requires_edge_count)
        """
        count = 0
        for pattern in self.patterns['edge_scan']:
            count += len(re.findall(pattern, code, re.IGNORECASE))
        
        if count == 0:
            return 0.0, False
        
        if graph_stats and 'num_edges' in graph_stats:
            # EScan(E, n) = |E| or |E|/n
            n = graph_stats.get('edge_scan_factor', 1)
            coeff = graph_stats['num_edges'] / n
            return float(coeff), True
        
        return float(count), True
    
    def _extract_frontier_scan(
        self,
        code: str,
        graph_stats: Optional[Dict[str, Any]]
    ) -> Tuple[float, bool]:
        """
        Extract frontier iteration feature.
        
        Returns:
            (coefficient, requires_diameter)
        """
        count = 0
        for pattern in self.patterns['frontier_scan']:
            count += len(re.findall(pattern, code, re.IGNORECASE))
        
        if count == 0:
            return 0.0, False
        
        if graph_stats and 'diameter' in graph_stats and 'num_edges' in graph_stats:
            # FScan(G, n) = sum_{t=1}^{D} |E_t|
            # Approximate: D * avg_edges_per_iteration
            diameter = graph_stats['diameter']
            avg_edges = graph_stats['num_edges'] / max(1, diameter)
            coeff = diameter * avg_edges
            return float(coeff), True
        
        return float(count), True
    
    def _extract_recursive_exp(
        self,
        code: str,
        graph_stats: Optional[Dict[str, Any]]
    ) -> Tuple[float, bool]:
        """
        Extract recursive expansion feature.
        
        Returns:
            (coefficient, requires_diameter_and_degree)
        """
        count = 0
        for pattern in self.patterns['recursive_exp']:
            count += len(re.findall(pattern, code, re.IGNORECASE))
        
        if count == 0:
            return 0.0, False
        
        if graph_stats and 'diameter' in graph_stats and 'avg_degree' in graph_stats:
            # RExp(G) = prod_{i=1}^{D} E[deg(v_i)]
            # Approximate: avg_degree ^ diameter
            diameter = min(graph_stats['diameter'], 10)  # Cap to avoid overflow
            avg_degree = graph_stats['avg_degree']
            coeff = avg_degree ** diameter
            return float(coeff), True
        
        return float(count), True
    
    def _extract_atomic_update(
        self,
        code: str,
        graph_stats: Optional[Dict[str, Any]]
    ) -> Tuple[float, bool]:
        """
        Extract atomic update feature.
        
        Returns:
            (coefficient, requires_skew)
        """
        count = 0
        for pattern in self.patterns['atomic_update']:
            count += len(re.findall(pattern, code, re.IGNORECASE))
        
        if count == 0:
            return 0.0, False
        
        if graph_stats and 'num_edges' in graph_stats and 'skew' in graph_stats:
            # Atom(G) = |E| * skew(G)
            coeff = graph_stats['num_edges'] * graph_stats['skew']
            return float(coeff), True
        
        return float(count), True
    
    def _extract_cross_partition(
        self,
        code: str,
        graph_stats: Optional[Dict[str, Any]],
        partition_stats: Optional[Dict[str, Any]]
    ) -> Tuple[float, bool]:
        """
        Extract cross-partition communication feature.
        
        Returns:
            (coefficient, requires_boundary_degree)
        """
        count = 0
        for pattern in self.patterns['cross_partition']:
            count += len(re.findall(pattern, code, re.IGNORECASE))
        
        if count == 0:
            return 0.0, False
        
        if partition_stats and 'boundary_degree_sum' in partition_stats:
            # Comm(G, F) = sum_{v in V_boundary} deg_boundary(v)
            coeff = partition_stats['boundary_degree_sum']
            return float(coeff), True
        
        if graph_stats and 'num_vertices' in graph_stats:
            # Rough estimate: boundary vertices * avg boundary degree
            boundary_ratio = 0.1  # Assume 10% boundary vertices
            avg_boundary_degree = graph_stats.get('avg_degree', 10) / 2
            coeff = graph_stats['num_vertices'] * boundary_ratio * avg_boundary_degree
            return float(coeff), True
        
        return float(count), True
    
    def extract_with_llm(
        self,
        source_code: str,
        llm_client: Any = None,
        graph_stats: Optional[Dict[str, Any]] = None,
        partition_stats: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract symbolic features using LLM-assisted pattern recognition.
        
        Args:
            source_code: Source code string
            llm_client: Optional LLM client for pattern recognition
            graph_stats: Graph statistics
            partition_stats: Partition statistics
            
        Returns:
            numpy array of symbolic features
        """
        if llm_client is None:
            # Fallback to regex-based extraction
            return self.extract(source_code, graph_stats, partition_stats)
        
        # Use LLM to identify patterns
        prompt = f"""
Analyze the following code and identify symbolic workload patterns:

Code:
{source_code[:5000]}  # Truncate for token limit

Identify these patterns and their frequencies:
1. Vertex scanning (vertex iteration)
2. Edge scanning (neighbor traversal)
3. Frontier iteration (worklist-driven loops)
4. Recursive expansion (recursive neighbor expansion)
5. Atomic updates (atomic writes, CAS)
6. Cross-partition communication (remote sends, mirror accesses)

Return as JSON with pattern names and counts.
"""
        
        try:
            response = llm_client.generate(prompt)
            pattern_counts = self._parse_llm_response(response)
            
            # Compute features with graph stats
            features = self._compute_features_from_counts(
                pattern_counts, graph_stats, partition_stats
            )
            return features
            
        except Exception as e:
            # Fallback to regex-based
            return self.extract(source_code, graph_stats, partition_stats)
    
    def _parse_llm_response(self, response: str) -> Dict[str, int]:
        """Parse LLM response to extract pattern counts."""
        # Simple JSON parsing (can be enhanced)
        import json
        try:
            return json.loads(response)
        except:
            return {}
    
    def _compute_features_from_counts(
        self,
        pattern_counts: Dict[str, int],
        graph_stats: Optional[Dict[str, Any]],
        partition_stats: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Compute symbolic features from pattern counts."""
        features = []
        
        # Map pattern names to extraction methods
        pattern_map = {
            'vertex_scan': (self._instantiate_vscan, graph_stats),
            'edge_scan': (self._instantiate_escan, graph_stats),
            'frontier_scan': (self._instantiate_fscan, graph_stats),
            'recursive_exp': (self._instantiate_rexp, graph_stats),
            'atomic_update': (self._instantiate_atom, graph_stats),
            'cross_partition': (self._instantiate_comm, partition_stats),
        }
        
        for pattern_name, (instantiate_fn, stats) in pattern_map.items():
            count = pattern_counts.get(pattern_name, 0)
            if count > 0 and stats:
                coeff, req = instantiate_fn(count, stats)
            else:
                coeff, req = float(count), count > 0
            
            features.append(coeff)
            features.append(1.0 if req else 0.0)
        
        return np.array(features, dtype=np.float64)
    
    def _instantiate_vscan(
        self,
        count: int,
        stats: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """Instantiate vertex scan with graph stats."""
        n = stats.get('vertex_scan_factor', 1)
        return stats['num_vertices'] / n, True
    
    def _instantiate_escan(
        self,
        count: int,
        stats: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """Instantiate edge scan with graph stats."""
        n = stats.get('edge_scan_factor', 1)
        return stats['num_edges'] / n, True
    
    def _instantiate_fscan(
        self,
        count: int,
        stats: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """Instantiate frontier scan with graph stats."""
        diameter = stats.get('diameter', 1)
        return diameter * stats.get('num_edges', 0) / max(1, diameter), True
    
    def _instantiate_rexp(
        self,
        count: int,
        stats: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """Instantiate recursive expansion with graph stats."""
        diameter = min(stats.get('diameter', 1), 10)
        avg_degree = stats.get('avg_degree', 2)
        return avg_degree ** diameter, True
    
    def _instantiate_atom(
        self,
        count: int,
        stats: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """Instantiate atomic update with graph stats."""
        return stats.get('num_edges', 0) * stats.get('skew', 1), True
    
    def _instantiate_comm(
        self,
        count: int,
        stats: Optional[Dict[str, Any]]
    ) -> Tuple[float, bool]:
        """Instantiate cross-partition communication with partition stats."""
        if stats and 'boundary_degree_sum' in stats:
            return stats['boundary_degree_sum'], True
        return float(count), True
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()
    
    def get_symbolic_templates(self) -> List[str]:
        """Return symbolic template descriptions."""
        return [
            'VScan(V, n): Vertex-linear work',
            'EScan(E, n): Edge-traversal work',
            'FScan(G, n): Round-sensitive propagation',
            'RExp(G): Branching search growth',
            'Atom(G): Contention and serialization',
            'Comm(G, F): Cross-partition communication',
        ]
