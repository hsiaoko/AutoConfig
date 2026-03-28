"""
Example: Feature extraction for Hybrid graph query tasks.

Demonstrates the three-stage feature extraction pipeline:
1. Static feature extraction from query code
2. Symbolic template matching
3. Graph-aware instantiation
"""

import numpy as np
import networkx as nx
from autoconfig.feature_extractor import (
    StaticFeatureExtractor,
    SymbolicFeatureExtractor,
    GraphPartitionExtractor,
    FeatureManager,
)


# Example query code: Subgraph isomorphism (from the paper)
SUBGRAPH_ISO_CODE = """
PEval(Fragment F, Context ctx):
  for v in F.Vertices():  // vertex traversal
    if !ctx.IsCandidate(v, p[0]): continue
    Match m
    m.Bind(P[0], v)
    Expand(m, 1, P, F, ctx)

Expand(Match& m, int level, Pattern P, Fragment F, Context ctx):
  if level == PatternSize: return
  for v in neighbor(m.Last()):  // edge traversal
    if !ctx.IsCandidate(v, P[level]) continue
    if !ctx.Consistent(m, v, P[level]) continue
    m.Bind(P[level], v)
    Expand(m, level + 1, P, F, ctx)  // recursive expansion
    m.Unbind(P[level])
    if ctx.IsMirror(v):  // cross-partition communication
      ctx.SendTo(context.Owner(v), ComputeMessage(v, context))
"""

# Example query code: PageRank
PAGERANK_CODE = """
PageRank(Graph G, int max_iter, float damping):
  // Initialize ranks
  for v in G.vertices():
    rank[v] = 1.0 / |V|
  
  // Iterative computation
  for iter in 1..max_iter:
    for v in G.vertices():  // vertex scan
      contribution = rank[v] / out_degree(v)
      for neighbor in G.out_edges(v):  // edge scan
        atomicAdd(new_rank[neighbor], contribution)  // atomic update
      new_rank[v] += damping * contribution
  
  // Synchronization barrier
  barrier()
  rank = new_rank
"""

# Example query code: BFS
BFS_CODE = """
BFS(Graph G, vertex source):
  worklist = [source]
  visited[source] = true
  level[source] = 0
  
  while !worklist.empty():  // frontier iteration
    next_worklist = []
    for v in worklist:
      for neighbor in G.neighbors(v):  // edge scan
        if !visited[neighbor]:
          visited[neighbor] = true
          level[neighbor] = level[v] + 1
          next_worklist.append(neighbor)
    worklist = next_worklist
"""


def example_static_extraction():
    """Example 1: Static feature extraction."""
    print("=" * 60)
    print("Example 1: Static Feature Extraction")
    print("=" * 60)
    
    extractor = StaticFeatureExtractor()
    
    # Extract from subgraph isomorphism code
    features = extractor.extract(SUBGRAPH_ISO_CODE)
    names = extractor.get_feature_names()
    
    print("\nSubgraph Isomorphism - Static Features:")
    for name, value in zip(names, features):
        print(f"  {name}: {value:.1f}")
    
    # Extract from PageRank code
    features_pr = extractor.extract(PAGERANK_CODE)
    print("\nPageRank - Static Features:")
    for name, value in zip(names, features_pr):
        print(f"  {name}: {value:.1f}")


def example_symbolic_extraction():
    """Example 2: Symbolic feature extraction with graph instantiation."""
    print("\n" + "=" * 60)
    print("Example 2: Symbolic Feature Extraction")
    print("=" * 60)
    
    extractor = SymbolicFeatureExtractor()
    
    # Create a sample graph
    graph = nx.erdos_renyi_graph(1000, 0.05)
    
    # Compute graph statistics
    graph_stats = {
        'num_vertices': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'diameter': 3,  # Approximate
        'avg_degree': np.mean([d for n, d in graph.degree()]),
        'max_degree': max(d for n, d in graph.degree()),
        'skew': 2.5,
    }
    
    # Extract symbolic features for BFS
    features = extractor.extract(BFS_CODE, graph_stats)
    names = extractor.get_feature_names()
    
    print("\nBFS - Symbolic Features (instantiated):")
    for i, (name, value) in enumerate(zip(names, features)):
        if i % 2 == 0:  # Coefficient
            print(f"  {name}: {value:.2f}")
        else:  # Requires flag
            print(f"  {name}: {'Yes' if value > 0 else 'No'}")


def example_graph_partition_extraction():
    """Example 3: Graph and partition feature extraction."""
    print("\n" + "=" * 60)
    print("Example 3: Graph & Partition Feature Extraction")
    print("=" * 60)
    
    extractor = GraphPartitionExtractor()
    
    # Create a sample graph
    graph = nx.erdos_renyi_graph(500, 0.05)
    
    # Create synthetic partitions
    n_partitions = 4
    nodes = list(graph.nodes())
    np.random.seed(42)
    np.random.shuffle(nodes)
    
    partitions = {}
    chunk_size = len(nodes) // n_partitions
    for i in range(n_partitions):
        start = i * chunk_size
        end = start + chunk_size if i < n_partitions - 1 else len(nodes)
        partitions[i] = nodes[start:end]
    
    # Extract features
    features = extractor.extract(graph, partitions)
    names = extractor.get_feature_names()
    
    print(f"\nGraph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Partitions: {n_partitions}")
    print("\nExtracted Features:")
    for name, value in zip(names, features):
        print(f"  {name}: {value:.4f}")
    
    # Get as dictionary
    stats_dict = extractor.get_graph_stats_dict(graph, partitions)
    print("\nGraph Stats Dictionary (for symbolic instantiation):")
    for key, value in stats_dict.items():
        print(f"  {key}: {value}")


def example_full_pipeline():
    """Example 4: Full three-stage feature extraction pipeline."""
    print("\n" + "=" * 60)
    print("Example 4: Full Feature Extraction Pipeline")
    print("=" * 60)
    
    manager = FeatureManager()
    
    # Create sample data
    graph = nx.erdos_renyi_graph(200, 0.1)
    
    # Create partitions
    nodes = list(graph.nodes())
    np.random.seed(42)
    np.random.shuffle(nodes)
    partitions = {
        0: nodes[:100],
        1: nodes[100:],
    }
    
    # Configuration
    config = {
        'memory_limit': 8192,
        'num_threads': 4,
        'cache_size': 1024,
        'batch_size': 1000,
        'io_buffer_size': 64,
        'num_workers': 2,
        'timeout': 300,
        'enable_index': True,
        'index_type': 'btree',
        'compression_enabled': False,
    }
    
    # Extract all features for subgraph isomorphism
    features = manager.extract_all(
        SUBGRAPH_ISO_CODE, graph, config, partitions
    )
    
    # Get feature groups
    groups = manager.get_feature_groups()
    dims = manager.get_feature_dimensions()
    
    print(f"\nTotal features: {len(features)}")
    print(f"Feature dimensions: {dims}")
    print(f"  - Static: {dims[0]}")
    print(f"  - Symbolic: {dims[1]}")
    print(f"  - Graph/Partition: {dims[2]}")
    print(f"  - Config: {dims[3]}")
    
    print("\nFeature Groups:")
    for group_name, feature_names in groups.items():
        print(f"\n  {group_name.upper()} ({len(feature_names)} features):")
        for name in feature_names[:5]:  # Show first 5
            print(f"    - {name}")
        if len(feature_names) > 5:
            print(f"    ... and {len(feature_names) - 5} more")


def example_compare_queries():
    """Example 5: Compare feature vectors for different queries."""
    print("\n" + "=" * 60)
    print("Example 5: Compare Query Feature Vectors")
    print("=" * 60)
    
    manager = FeatureManager()
    
    # Create common graph and config
    graph = nx.erdos_renyi_graph(300, 0.05)
    config = {
        'memory_limit': 8192,
        'num_threads': 4,
        'cache_size': 1024,
        'batch_size': 1000,
        'io_buffer_size': 64,
        'num_workers': 2,
        'timeout': 300,
        'enable_index': True,
        'index_type': 'btree',
        'compression_enabled': False,
    }
    
    # Extract features for different queries
    queries = {
        'Subgraph Iso': SUBGRAPH_ISO_CODE,
        'PageRank': PAGERANK_CODE,
        'BFS': BFS_CODE,
    }
    
    print("\nComparing static features across queries:")
    extractor = StaticFeatureExtractor()
    names = extractor.get_feature_names()
    
    print(f"{'Feature':<30} {'Subgraph':<10} {'PageRank':<10} {'BFS':<10}")
    print("-" * 60)
    
    for i, name in enumerate(names):
        values = []
        for query_name, code in queries.items():
            features = extractor.extract(code)
            values.append(f"{features[i]:.1f}")
        print(f"{name:<30} {values[0]:<10} {values[1]:<10} {values[2]:<10}")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("# Hybrid Query Feature Extraction Examples")
    print("#" * 60)
    
    example_static_extraction()
    example_symbolic_extraction()
    example_graph_partition_extraction()
    example_full_pipeline()
    example_compare_queries()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
