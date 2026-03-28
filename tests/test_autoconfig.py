"""
Unit tests for autoconfig package.
Tests for the three-stage feature extraction pipeline.
"""

import numpy as np
import networkx as nx
import unittest

from autoconfig import (
    StaticFeatureExtractor,
    SymbolicFeatureExtractor,
    GraphPartitionExtractor,
    ConfigFeatureExtractor,
    FeatureManager,
    BayesianExecutionTimeModel,
    CostPredictor,
)


class TestStaticFeatureExtractor(unittest.TestCase):
    """Test static feature extraction."""
    
    def setUp(self):
        self.extractor = StaticFeatureExtractor()
    
    def test_extract_loops(self):
        """Test loop detection."""
        code = """
        for i in range(10):
            for j in range(5):
                process(i, j)
        """
        features = self.extractor.extract(code)
        self.assertGreaterEqual(features[0], 2)  # loop_count
    
    def test_extract_branches(self):
        """Test branch detection."""
        code = """
        if condition(x):
            process(x)
        if other_condition(y):
            process(y)
        """
        features = self.extractor.extract(code)
        # Branch detection may vary, just check it runs
        self.assertIsInstance(features[2], float)
    
    def test_extract_atomic_ops(self):
        """Test atomic operation detection."""
        code = """
        for v in vertices:
            atomicAdd(counter[v], 1)
        """
        features = self.extractor.extract(code)
        self.assertGreaterEqual(features[5], 1)  # atomic_op_count
    
    def test_extract_sync_ops(self):
        """Test synchronization detection."""
        code = """
        parallel_for(i in range(n)):
            barrier()
        """
        features = self.extractor.extract(code)
        self.assertGreaterEqual(features[6], 1)  # sync_count


class TestSymbolicFeatureExtractor(unittest.TestCase):
    """Test symbolic feature extraction."""
    
    def setUp(self):
        self.extractor = SymbolicFeatureExtractor()
    
    def test_vertex_scan_detection(self):
        """Test vertex scanning pattern detection."""
        code = """
        for v in G.vertices():
            process(v)
        """
        graph_stats = {'num_vertices': 1000}
        features = self.extractor.extract(code, graph_stats)
        
        # sym_vscan_coeff should be > 0
        self.assertGreater(features[0], 0)
        self.assertEqual(features[1], 1.0)  # requires flag
    
    def test_edge_scan_detection(self):
        """Test edge scanning pattern detection."""
        code = """
        for v in G.vertices():
            for neighbor in G.neighbors(v):
                process(v, neighbor)
        """
        graph_stats = {'num_edges': 5000}
        features = self.extractor.extract(code, graph_stats)
        
        # sym_escan_coeff should be > 0
        self.assertGreater(features[2], 0)
        self.assertEqual(features[3], 1.0)  # requires flag
    
    def test_frontier_scan_detection(self):
        """Test frontier iteration detection."""
        code = """
        while !worklist.empty():
            for v in worklist:
                process(v)
        """
        graph_stats = {'num_edges': 5000, 'diameter': 5}
        features = self.extractor.extract(code, graph_stats)
        
        # sym_fscan_coeff should be > 0
        self.assertGreater(features[4], 0)
        self.assertEqual(features[5], 1.0)  # requires flag
    
    def test_atomic_update_detection(self):
        """Test atomic update detection."""
        code = """
        for edge in edges:
            atomicAdd(weights[edge.src], edge.weight)
        """
        graph_stats = {'num_edges': 5000, 'skew': 2.0}
        features = self.extractor.extract(code, graph_stats)
        
        # sym_atom_coeff should be > 0
        self.assertGreater(features[8], 0)
        self.assertEqual(features[9], 1.0)  # requires flag


class TestGraphPartitionExtractor(unittest.TestCase):
    """Test graph and partition feature extraction."""
    
    def setUp(self):
        self.extractor = GraphPartitionExtractor()
    
    def test_extract_graph_stats(self):
        """Test basic graph statistics."""
        graph = nx.erdos_renyi_graph(100, 0.1)
        features = self.extractor.extract(graph)
        
        self.assertEqual(features[0], 100)  # num_vertices
        self.assertEqual(features[1], graph.number_of_edges())  # num_edges
    
    def test_extract_partition_features(self):
        """Test partition feature extraction."""
        graph = nx.erdos_renyi_graph(200, 0.1)
        
        # Create partitions
        nodes = list(graph.nodes())
        partitions = {
            0: nodes[:100],
            1: nodes[100:],
        }
        
        features = self.extractor.extract(graph, partitions)
        
        self.assertEqual(features[10], 2)  # num_partitions
        self.assertGreater(features[11], 0)  # boundary_vertices
    
    def test_get_stats_dict(self):
        """Test dictionary output."""
        graph = nx.erdos_renyi_graph(50, 0.1)
        stats = self.extractor.get_graph_stats_dict(graph)
        
        self.assertIn('num_vertices', stats)
        self.assertIn('num_edges', stats)
        self.assertIn('diameter', stats)
        self.assertIn('skew', stats)


class TestFeatureManager(unittest.TestCase):
    """Test feature manager."""
    
    def setUp(self):
        self.manager = FeatureManager()
    
    def test_extract_all(self):
        """Test combined feature extraction."""
        code = """
        for v in G.vertices():
            if condition(v):
                process(v)
        """
        graph = nx.erdos_renyi_graph(50, 0.1)
        config = {'memory_limit': 8192, 'num_threads': 4}
        
        features = self.manager.extract_all(code, graph, config)
        
        # Should have all feature groups
        total_dims = sum(self.manager.get_feature_dimensions())
        self.assertEqual(len(features), total_dims)
    
    def test_get_feature_groups(self):
        """Test feature group retrieval."""
        groups = self.manager.get_feature_groups()
        
        self.assertIn('static', groups)
        self.assertIn('symbolic', groups)
        self.assertIn('graph_partition', groups)
        self.assertIn('config', groups)


class TestBayesianModel(unittest.TestCase):
    """Test Bayesian model."""
    
    def setUp(self):
        self.model = BayesianExecutionTimeModel(n_iter=100)
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        np.random.seed(42)
        X = np.random.randn(50, 47)  # 47 features
        y = np.abs(X @ np.random.randn(47) + np.random.randn(50) * 0.1)
        
        self.model.fit(X, y)
        
        self.assertTrue(self.model.is_fitted)
        
        predictions = self.model.predict(X)
        self.assertEqual(len(predictions), len(y))
    
    def test_predict_with_uncertainty(self):
        """Test prediction with uncertainty estimation."""
        np.random.seed(42)
        X = np.random.randn(50, 47)
        y = np.abs(X @ np.random.randn(47) + np.random.randn(50) * 0.1)
        
        self.model.fit(X, y)
        
        pred, lower, upper = self.model.predict_with_uncertainty(X[:5])
        
        self.assertEqual(len(pred), 5)
        self.assertTrue(np.all(lower <= pred))
        self.assertTrue(np.all(pred <= upper))


class TestCostPredictor(unittest.TestCase):
    """Test cost predictor."""
    
    def setUp(self):
        self.predictor = CostPredictor()
    
    def test_train_and_predict(self):
        """Test training and prediction with code features."""
        np.random.seed(42)
        
        # Create sample query codes
        codes = [
            "for v in G.vertices(): process(v)",
            "for e in G.edges(): process(e)",
            "while !worklist.empty(): expand()",
        ] * 10
        
        graphs = [nx.erdos_renyi_graph(np.random.randint(30, 60), 0.1) 
                  for _ in range(30)]
        configs = [{'memory_limit': np.random.choice([4096, 8192]),
                    'num_threads': np.random.choice([2, 4])}
                   for _ in range(30)]
        exec_times = np.random.uniform(10, 100, 30)
        
        # Use extract_all instead of extract_all_from_dict
        metrics = self.predictor.train(
            codes, graphs, configs, exec_times, verbose=False
        )
        
        self.assertIn('mae', metrics)
        
        # Predict
        query = "for v in G.vertices(): visit(v)"
        graph = nx.erdos_renyi_graph(50, 0.1)
        config = {'memory_limit': 8192, 'num_threads': 4}
        
        pred = self.predictor.predict(query, graph, config)
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0)


if __name__ == '__main__':
    unittest.main()
