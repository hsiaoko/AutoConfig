"""
Generate sample graph data for testing.
"""

import numpy as np
import networkx as nx
from pathlib import Path


def generate_erdos_renyi(n_nodes, edge_prob, output_file):
    """Generate Erdos-Renyi random graph."""
    np.random.seed(42)
    graph = nx.erdos_renyi_graph(n_nodes, edge_prob)
    
    with open(output_file, 'w') as f:
        f.write("src,dst\n")
        for u, v in graph.edges():
            f.write(f"{u},{v}\n")
    
    print(f"Generated Erdos-Renyi graph: {n_nodes} nodes, {graph.number_of_edges()} edges")
    print(f"  Output: {output_file}")


def generate_power_law(n_nodes, m_edges, output_file):
    """Generate Barabasi-Albert preferential attachment graph (power-law degree distribution)."""
    np.random.seed(42)
    graph = nx.barabasi_albert_graph(n_nodes, m_edges)
    
    with open(output_file, 'w') as f:
        f.write("src,dst\n")
        for u, v in graph.edges():
            f.write(f"{u},{v}\n")
    
    print(f"Generated Power-Law graph: {n_nodes} nodes, {graph.number_of_edges()} edges")
    print(f"  Output: {output_file}")


def generate_small_world(n_nodes, k_neighbors, p_rewire, output_file):
    """Generate Watts-Strogatz small-world graph."""
    np.random.seed(42)
    graph = nx.watts_strogatz_graph(n_nodes, k_neighbors, p_rewire)
    
    with open(output_file, 'w') as f:
        f.write("src,dst\n")
        for u, v in graph.edges():
            f.write(f"{u},{v}\n")
    
    print(f"Generated Small-World graph: {n_nodes} nodes, {graph.number_of_edges()} edges")
    print(f"  Output: {output_file}")


def generate_partitioned_graph(base_graph, n_partitions, output_dir):
    """Generate partitioned graph from base graph."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nodes = list(base_graph.nodes())
    np.random.seed(42)
    np.random.shuffle(nodes)
    
    # Partition nodes
    partition_size = len(nodes) // n_partitions
    partitions = []
    for i in range(n_partitions):
        start = i * partition_size
        end = start + partition_size if i < n_partitions - 1 else len(nodes)
        partitions.append(set(nodes[start:end]))
    
    # Create node to partition mapping
    node_to_partition = {}
    for i, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = i
    
    # Extract edges for each partition
    partition_edges = [[] for _ in range(n_partitions)]
    for u, v in base_graph.edges():
        p_u = node_to_partition.get(u, 0)
        p_v = node_to_partition.get(v, 0)
        # Assign edge to partition of source node
        partition_edges[p_u].append((u, v))
    
    # Write partition files
    for i, edges in enumerate(partition_edges):
        output_file = output_dir / f"partition_{i}.csv"
        with open(output_file, 'w') as f:
            f.write("src,dst\n")
            for u, v in edges:
                f.write(f"{u},{v}\n")
        print(f"  Partition {i}: {len(edges)} edges -> {output_file}")
    
    print(f"Generated partitioned graph: {n_partitions} partitions")
    print(f"  Output: {output_dir}")


def main():
    """Generate all sample graphs."""
    data_dir = Path('data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Sample Graph Data")
    print("=" * 60)
    
    # Small graphs (for quick testing)
    print("\n--- Small Graphs ---")
    generate_erdos_renyi(100, 0.05, data_dir / 'graph_small_er.csv')
    generate_power_law(100, 3, data_dir / 'graph_small_pl.csv')
    generate_small_world(100, 4, 0.1, data_dir / 'graph_small_sw.csv')
    
    # Medium graphs (for realistic testing)
    print("\n--- Medium Graphs ---")
    generate_erdos_renyi(1000, 0.01, data_dir / 'graph_medium_er.csv')
    generate_power_law(1000, 5, data_dir / 'graph_medium_pl.csv')
    generate_small_world(1000, 10, 0.05, data_dir / 'graph_medium_sw.csv')
    
    # Large graphs (for stress testing)
    print("\n--- Large Graphs ---")
    generate_erdos_renyi(5000, 0.005, data_dir / 'graph_large_er.csv')
    generate_power_law(5000, 10, data_dir / 'graph_large_pl.csv')
    
    # Partitioned graphs
    print("\n--- Partitioned Graphs ---")
    
    # Small partitioned
    graph_small = nx.erdos_renyi_graph(300, 0.05)
    generate_partitioned_graph(graph_small, 3, data_dir / 'graph_small_partitioned')
    
    # Medium partitioned
    graph_medium = nx.powerlaw_cluster_graph(1000, 5, 0.05)
    generate_partitioned_graph(graph_medium, 4, data_dir / 'graph_medium_partitioned')
    
    print("\n" + "=" * 60)
    print("Sample graph generation complete!")
    print("=" * 60)
    
    # Print summary
    print("\nGenerated files:")
    for f in sorted(data_dir.glob('*.csv')):
        print(f"  {f}")
    for d in sorted(data_dir.glob('*/')):
        print(f"  {d}/ (partitioned)")


if __name__ == '__main__':
    main()
