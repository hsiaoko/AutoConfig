"""
Feature Merge Tool

Merges query, graph, and configuration features into a single feature vector.
Instantiates symbolic features with actual graph statistics.

Usage:
    python -m autoconfig.utils.feature_merger \
        --query out/query_features.yaml \
        --graph out/graph_features.yaml \
        --config out/config_features.yaml \
        --output out/merged_features.yaml
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

from ..feature_extractor.config_extractor import DEFAULT_CONFIG_BATCH_SIZE


class FeatureMerger:
    """
    Merges features from query, graph, and configuration.
    
    Process:
    1. Load query features (with placeholders)
    2. Load graph features (with actual statistics)
    3. Load config features (multiple configurations)
    4. Instantiate symbolic features with graph stats
    5. Merge into complete feature vectors

    **Layout (53 dimensions, fixed order):** ``price``, ``time``, ``cost``, then static (8),
    symbolic (12), graph+partition (17), config (13, including ``conf_price`` from catalog).
    At merge time: ``price`` and ``conf_price`` come from the config catalog, ``time`` and
    ``cost`` are set to **0** (fill ``time`` e.g. from GridGraph ``STATS`` downstream).
    """
    
    _STATIC = [
        'static_loop_count',
        'static_max_loop_depth',
        'static_branch_count',
        'static_variable_count',
        'static_recursion_count',
        'static_atomic_op_count',
        'static_sync_count',
        'static_explicit_parallel_flag',
    ]
    _SYM = [
        'sym_vscan_coeff', 'sym_vscan_requires', 'sym_escan_coeff', 'sym_escan_requires',
        'sym_fscan_coeff', 'sym_fscan_requires', 'sym_rexp_coeff', 'sym_rexp_requires',
        'sym_atom_coeff', 'sym_atom_requires', 'sym_comm_coeff', 'sym_comm_requires',
    ]
    _GRAPH = [
        'graph_num_vertices', 'graph_num_edges', 'graph_diameter', 'graph_avg_degree',
        'graph_max_degree', 'graph_min_degree', 'graph_degree_std', 'graph_skew',
        'graph_clustering_coeff', 'graph_num_components', 'partition_num_partitions',
        'partition_boundary_vertices', 'partition_boundary_degree_sum',
        'partition_avg_partition_size', 'partition_size_std', 'partition_edge_cut_ratio',
        'partition_balance',
    ]
    _CONFIG_NO_PRICE = [
        'conf_memory_limit', 'conf_num_threads', 'conf_cache_size', 'conf_batch_size',
        'conf_io_buffer_size', 'conf_num_workers', 'conf_timeout', 'conf_enable_index',
        'conf_index_type', 'conf_compression_enabled', 'conf_grid_size', 'conf_block_size',
    ]
    _CONF_PRICE = 'conf_price'
    
    def __init__(self) -> None:
        self.feature_order: List[str] = (
            ['price', 'time', 'cost']
            + self._STATIC
            + self._SYM
            + self._GRAPH
            + self._CONFIG_NO_PRICE
            + [self._CONF_PRICE]
        )
        if len(self.feature_order) != 53:
            raise RuntimeError("internal: expected 53 feature names")
    
    def _feature_group_counts(self) -> Dict[str, int]:
        return {
            "price": 1,
            "time": 1,
            "cost": 1,
            "static": 8,
            "symbolic": 12,
            "graph": 17,
            "config": 13,
        }

    def load_yaml(self, filepath: str) -> Dict[str, Any]:
        """Load YAML file (handling numpy types)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            # Use unsafe loader to handle numpy types
            return yaml.unsafe_load(f)

    def graph_stats_for_symbolic_instantiation(
        self,
        graph_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Map graph_features YAML to the statistic names used in symbolic templates.

        Uses partition count as n in VScan/EScan when present (|V|/n, |E|/n).
        """
        gf = graph_data.get('graph_features', {})
        npart = max(1, int(gf.get('partition', {}).get('num_partitions', 1)))
        # Allow top-level `diameter` (mirror) or legacy/flat files missing `structure`
        raw_d = gf.get('structure', {}).get('diameter', gf.get('diameter', 0))
        diam = int(raw_d) if raw_d is not None else 0
        if diam <= 0:
            diam = 1
        skew = float(gf.get('degree', {}).get('skew', 1.0))
        if skew == 0.0:
            skew = 1.0
        return {
            'num_vertices': float(gf.get('basic', {}).get('num_vertices', 0)),
            'num_edges': float(gf.get('basic', {}).get('num_edges', 0)),
            'diameter': float(diam),
            'avg_degree': float(gf.get('degree', {}).get('avg', 0)),
            'skew': skew,
            'boundary_degree_sum': float(gf.get('partition', {}).get('boundary_degree_sum', 0)),
            'vertex_scan_factor': float(npart),
            'edge_scan_factor': float(npart),
        }

    def _instantiate_symbolic_from_families(
        self,
        families: Dict[str, Any],
        stats: Dict[str, float],
    ) -> Dict[str, float]:
        """Turn template families + graph stats into the legacy 12 float names."""
        nv = stats.get('num_vertices', 0.0)
        ne = stats.get('num_edges', 0.0)
        n_v = max(1.0, stats.get('vertex_scan_factor', 1.0))
        n_e = max(1.0, stats.get('edge_scan_factor', 1.0))
        d_raw = int(stats.get('diameter', 1))
        d_eff = min(max(1, d_raw), 10)
        avg_deg = stats.get('avg_degree', 0.0)
        skew = stats.get('skew', 1.0)
        bnd = stats.get('boundary_degree_sum', 0.0)

        out: Dict[str, float] = {}

        def pair(fid: str, coeff: float, active: bool) -> None:
            out[f'sym_{fid}_coeff'] = float(coeff) if active else 0.0
            out[f'sym_{fid}_requires'] = 1.0 if active else 0.0

        fv = families.get('vscan', {})
        pair('vscan', nv / n_v, bool(fv.get('detected')))

        fe = families.get('escan', {})
        pair('escan', ne / n_e, bool(fe.get('detected')))

        ff = families.get('fscan', {})
        # Approximate sum_t |E_t| by |E| when per-round edges are unavailable
        pair('fscan', float(ne), bool(ff.get('detected')))

        fr = families.get('rexp', {})
        rexp_val = (avg_deg ** d_eff) if avg_deg > 0 else 0.0
        pair('rexp', rexp_val, bool(fr.get('detected')))

        fa = families.get('atom', {})
        pair('atom', ne * skew, bool(fa.get('detected')))

        fc = families.get('comm', {})
        pair('comm', bnd, bool(fc.get('detected')))

        return out

    def symbolic_family_instantiation_detail(
        self,
        families: Dict[str, Any],
        stats: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Per-family: graph-instantiated values are always reported (``graph_instance_value``,
        ``instance_expr``).         The ``sym_*_coeff`` / ``sym_*_requires`` fields match the
        merged feature vector (after leading ``cost``): they stay zero unless ``detected`` is true in the query YAML
        (static/symbolic extraction found that pattern in source code).
        """
        nv = stats.get('num_vertices', 0.0)
        ne = stats.get('num_edges', 0.0)
        n_v = max(1.0, stats.get('vertex_scan_factor', 1.0))
        n_e = max(1.0, stats.get('edge_scan_factor', 1.0))
        d_raw = int(stats.get('diameter', 1))
        d_eff = min(max(1, d_raw), 10)
        avg_deg = stats.get('avg_degree', 0.0)
        skew = stats.get('skew', 1.0)
        bnd = stats.get('boundary_degree_sum', 0.0)

        def sub(fid: str, detected: bool, graph_value: float, expr: str) -> Dict[str, Any]:
            return {
                'detected': bool(detected),
                'graph_instance_value': float(graph_value),
                'instance_expr': expr,
                'sym_' + fid + '_coeff': float(graph_value) if detected else 0.0,
                'sym_' + fid + '_requires': 1.0 if detected else 0.0,
            }

        out: Dict[str, Any] = {}
        fv = families.get('vscan', {})
        out['vscan'] = sub(
            'vscan',
            bool(fv.get('detected')),
            nv / n_v,
            f'|V|/n = {int(nv)}/{int(n_v)}',
        )
        fe = families.get('escan', {})
        out['escan'] = sub(
            'escan',
            bool(fe.get('detected')),
            ne / n_e,
            f'|E|/n = {int(ne)}/{int(n_e)}',
        )
        ff = families.get('fscan', {})
        out['fscan'] = sub(
            'fscan',
            bool(ff.get('detected')),
            float(ne),
            f'|E| (proxy for sum_t |E_t|) = {int(ne)}',
        )
        fr = families.get('rexp', {})
        rexp_val = (avg_deg ** d_eff) if avg_deg > 0 else 0.0
        out['rexp'] = sub(
            'rexp',
            bool(fr.get('detected')),
            rexp_val,
            f'E[deg]^{d_eff} = {avg_deg}^{d_eff} (d={d_raw} clamped 1..10)',
        )
        fa = families.get('atom', {})
        out['atom'] = sub(
            'atom',
            bool(fa.get('detected')),
            ne * skew,
            f'|E|*skew = {int(ne)}*{skew:.6g}',
        )
        fc = families.get('comm', {})
        out['comm'] = sub(
            'comm',
            bool(fc.get('detected')),
            bnd,
            f'boundary_degree_sum = {bnd:.6g}',
        )
        return out

    def _is_legacy_symbolic_flat(self, symbolic: Dict[str, Any]) -> bool:
        return 'sym_vscan_coeff' in symbolic and 'families' not in symbolic

    def instantiate_symbolic(
        self,
        symbolic_block: Dict[str, Any],
        graph_stats: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Instantiate symbolic features with graph statistics.

        Accepts either:
        - format_version 2 with ``families`` (templates + formulas in query YAML), or
        - legacy flat dict with sym_*_coeff / sym_*_requires placeholders.
        """
        if symbolic_block.get('format_version') == 2 or 'families' in symbolic_block:
            families = symbolic_block.get('families', {})
            return self._instantiate_symbolic_from_families(families, graph_stats)

        if self._is_legacy_symbolic_flat(symbolic_block):
            instantiated = dict(symbolic_block)

            if symbolic_block.get('sym_vscan_requires', 0) > 0:
                n = max(1.0, graph_stats.get('vertex_scan_factor', 1.0))
                instantiated['sym_vscan_coeff'] = graph_stats.get('num_vertices', 0) / n

            if symbolic_block.get('sym_escan_requires', 0) > 0:
                n = max(1.0, graph_stats.get('edge_scan_factor', 1.0))
                instantiated['sym_escan_coeff'] = graph_stats.get('num_edges', 0) / n

            if symbolic_block.get('sym_fscan_requires', 0) > 0:
                instantiated['sym_fscan_coeff'] = graph_stats.get('num_edges', 0)

            if symbolic_block.get('sym_rexp_requires', 0) > 0:
                avg_degree = graph_stats.get('avg_degree', 2)
                diameter = min(int(graph_stats.get('diameter', 3)), 10)
                instantiated['sym_rexp_coeff'] = avg_degree ** max(1, diameter)

            if symbolic_block.get('sym_atom_requires', 0) > 0:
                instantiated['sym_atom_coeff'] = (
                    graph_stats.get('num_edges', 0) * graph_stats.get('skew', 1)
                )

            if symbolic_block.get('sym_comm_requires', 0) > 0:
                instantiated['sym_comm_coeff'] = graph_stats.get('boundary_degree_sum', 0)

            return instantiated

        # Empty or unknown: zeros
        return {
            'sym_vscan_coeff': 0.0,
            'sym_vscan_requires': 0.0,
            'sym_escan_coeff': 0.0,
            'sym_escan_requires': 0.0,
            'sym_fscan_coeff': 0.0,
            'sym_fscan_requires': 0.0,
            'sym_rexp_coeff': 0.0,
            'sym_rexp_requires': 0.0,
            'sym_atom_coeff': 0.0,
            'sym_atom_requires': 0.0,
            'sym_comm_coeff': 0.0,
            'sym_comm_requires': 0.0,
        }
    
    def extract_graph_features(
        self,
        graph_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract graph features from loaded graph YAML."""
        gf = graph_data.get('graph_features', {})
        diam = gf.get('structure', {}).get('diameter', gf.get('diameter', 0))
        
        return {
            'graph_num_vertices': gf.get('basic', {}).get('num_vertices', 0),
            'graph_num_edges': gf.get('basic', {}).get('num_edges', 0),
            'graph_diameter': diam,
            'graph_avg_degree': gf.get('degree', {}).get('avg', 0),
            'graph_max_degree': gf.get('degree', {}).get('max', 0),
            'graph_min_degree': gf.get('degree', {}).get('min', 0),
            'graph_degree_std': gf.get('degree', {}).get('std', 0),
            'graph_skew': gf.get('degree', {}).get('skew', 0),
            'graph_clustering_coeff': gf.get('structure', {}).get('clustering_coeff', 0),
            'graph_num_components': gf.get('structure', {}).get('num_components', 1),
            'partition_num_partitions': gf.get('partition', {}).get('num_partitions', 1),
            'partition_boundary_vertices': gf.get('partition', {}).get('boundary_vertices', 0),
            'partition_boundary_degree_sum': gf.get('partition', {}).get('boundary_degree_sum', 0),
            'partition_avg_partition_size': gf.get('partition', {}).get('avg_partition_size', 0),
            'partition_size_std': gf.get('partition', {}).get('partition_size_std', 0),
            'partition_edge_cut_ratio': gf.get('partition', {}).get('edge_cut_ratio', 0),
            'partition_balance': gf.get('partition', {}).get('balance', 1),
        }
    
    def _catalog_price(self, config_data: Dict[str, Any]) -> float:
        """``catalog`` often starts with ``{ price: <float> }``; this must be preserved in merge."""
        for item in config_data.get("catalog") or []:
            if isinstance(item, dict) and "price" in item:
                return float(item["price"])
        return 0.0

    def config_scalars(self, config_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Flat scalars from the first ``configuration`` row (and related catalog/metadata), for
        merged YAMLs — same idea as :meth:`graph_stats_for_symbolic_instantiation` / ``graph_scalars``.
        """
        out: Dict[str, float] = {}
        out["price"] = self._catalog_price(config_data)
        cfg0: Dict[str, Any] = {}
        confs = config_data.get("configurations") or []
        if confs and isinstance(confs[0], dict):
            cfg0 = dict(confs[0])
        if "k" in cfg0:
            out["k"] = float(int(cfg0.get("k") or 1))
        if cfg0.get("node_id") is not None:
            out["node_id"] = float(cfg0.get("node_id", 0))
        res: Dict[str, Any] = dict(cfg0.get("resource") or {})
        if not res:
            for item in config_data.get("catalog") or []:
                if not isinstance(item, dict):
                    continue
                if any(
                    k in item
                    for k in ("cpu_cores", "memory_gb", "storage_gb", "num_gpus")
                ):
                    res = {k: v for k, v in item.items() if k != "price"}
                    break
        for key in (
            "cpu_cores",
            "memory_gb",
            "storage_gb",
            "num_gpus",
            "grid_size",
            "block_size",
            "gpu_memory_gb",
        ):
            v = res.get(key)
            if v is not None:
                out[key] = float(v)
        meta = config_data.get("metadata") or {}
        if meta.get("sample_index") is not None:
            out["sample_index"] = float(meta["sample_index"])
        cp = meta.get("cloud_pricing") or {}
        for key in ("list_price_usd_per_hour", "estimated_monthly_usd_730h"):
            v = cp.get(key)
            if v is not None:
                out[key] = float(v)
        return out

    def extract_config_features(
        self,
        config_data: Dict[str, Any]
    ) -> List[Dict[str, float]]:
        """Extract config features from loaded config YAML."""
        price = self._catalog_price(config_data)
        config_features = config_data.get('config_features', [])
        if not config_features:
            configs = config_data.get("configurations") or []
            if not configs:
                rows = [self._config_to_features({})]
            else:
                rows = [self._config_to_features(c) for c in configs]
        else:
            rows = [self._config_to_features(cf) for cf in config_features]
        for r in rows:
            r["conf_price"] = price
        return rows
    
    def _config_to_features(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Convert single config to feature dictionary."""
        if "k" in config:
            k = int(config.get("k") or 1)
        else:
            # Single-node rows may use `node_id` instead of replica count k
            k = 1
        resource = config.get("resource", config.get("s", config.get("conf_per_instance", {})))

        return {
            "conf_memory_limit": k * resource.get("memory_gb", 8) * 1024,  # MB
            "conf_num_threads": k * resource.get("cpu_cores", 4),
            "conf_cache_size": k * resource.get("memory_gb", 8) * 1024 // 8,
            "conf_batch_size": float(
                resource.get("batch_size", DEFAULT_CONFIG_BATCH_SIZE)
            ),
            "conf_io_buffer_size": 64,
            "conf_num_workers": k * max(1, resource.get("cpu_cores", 4) // 4),
            "conf_timeout": 300,
            "conf_enable_index": 1.0,
            "conf_index_type": 1.0,
            "conf_compression_enabled": 0.0,
            "conf_grid_size": float(resource.get("grid_size", 0) or 0),
            "conf_block_size": float(resource.get("block_size", 0) or 0),
        }
    
    def merge(
        self,
        query_features: Dict[str, Any],
        graph_features: Dict[str, Any],
        config_features: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Merge all features into feature vectors.
        
        Args:
            query_features: Query feature dictionary
            graph_features: Graph feature dictionary
            config_features: List of config feature dictionaries
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Extract static features
        static = query_features.get('query_features', {}).get('static', {})
        
        # Extract and instantiate symbolic features (templates -> numbers using graph YAML)
        symbolic = query_features.get('query_features', {}).get('symbolic', {})
        graph_stats = self.graph_stats_for_symbolic_instantiation(graph_features)
        symbolic_instantiated = self.instantiate_symbolic(symbolic, graph_stats)
        graph_block = self.extract_graph_features(graph_features)
        
        # Build merged features for each configuration
        feature_vectors = []
        
        for cf in config_features:
            merged: Dict[str, float] = {}
            merged.update(static)
            merged.update(symbolic_instantiated)
            merged.update(graph_block)
            merged.update(graph_stats)
            catalog_price = float(cf.get("conf_price", 0.0))
            merged['price'] = catalog_price
            merged['time'] = 0.0
            merged['cost'] = 0.0
            merged.update(cf)

            # Create feature vector in correct order
            vector = []
            for name in self.feature_order:
                vector.append(float(merged.get(name, 0.0)))

            feature_vectors.append(vector)
        
        return (
            np.array(feature_vectors, dtype=np.float64),
            self.feature_order.copy(),
        )
    
    def merge_all(
        self,
        query_file: str,
        graph_file: str,
        config_file: str
    ) -> Dict[str, Any]:
        """
        Load and merge all features from files.
        
        Args:
            query_file: Path to query features YAML
            graph_file: Path to graph features YAML
            config_file: Path to config features YAML
            
        Returns:
            Complete merged feature data
        """
        # Load all files
        query_data = self.load_yaml(query_file)
        graph_data = self.load_yaml(graph_file)
        config_data = self.load_yaml(config_file)
        
        # Extract features
        config_features = self.extract_config_features(config_data)
        
        # Merge
        feature_matrix, feature_names = self.merge(
            query_data,
            graph_data,
            config_features
        )
        graph_stats = self.graph_stats_for_symbolic_instantiation(graph_data)
        sym_block = query_data.get('query_features', {}).get('symbolic', {})
        sym_families = sym_block.get('families', {})
        symbolic_detail: Dict[str, Any] = {}
        if sym_families:
            symbolic_detail = self.symbolic_family_instantiation_detail(
                sym_families, graph_stats
            )
        
        # Build result
        result = {
            'feature_matrix': feature_matrix.tolist(),
            'feature_names': feature_names,
            'config_scalars': self.config_scalars(config_data),
            'graph_scalars': graph_stats,
            'symbolic_families_instantiation': symbolic_detail,
            'metadata': {
                'num_samples': len(config_features),
                'num_features': len(feature_names),
                'query_file': str(query_file),
                'graph_file': str(graph_file),
                'config_file': str(config_file),
            },
            'feature_groups': self._feature_group_counts(),
        }
        
        return result
    
    def save_to_yaml(
        self,
        data: Dict[str, Any],
        output_path: str
    ):
        """Save merged features to YAML."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Merge query, graph, and configuration features'
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        required=True,
        help='Query features YAML file'
    )
    parser.add_argument(
        '--graph', '-g',
        type=str,
        required=True,
        help='Graph features YAML file'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Configuration features YAML file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='out/merged_features.yaml',
        help='Output merged features YAML file'
    )
    
    args = parser.parse_args()
    
    # Merge features
    merger = FeatureMerger()
    result = merger.merge_all(args.query, args.graph, args.config)
    
    # Save to YAML
    merger.save_to_yaml(result, args.output)
    
    # Print summary
    print("Features merged successfully:")
    print(f"  Query: {args.query}")
    print(f"  Graph: {args.graph}")
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output}")
    print(f"  Feature matrix: {result['metadata']['num_samples']} samples × {result['metadata']['num_features']} features")
    print("\nFeature groups:")
    for group, count in result['feature_groups'].items():
        print(f"  {group}: {count}")


if __name__ == '__main__':
    main()
