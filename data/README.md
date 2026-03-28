# Sample Graph Data

This directory contains sample graph data for testing AutoConfig.

## Graph Types

### Small Graphs (~100 nodes, ~200-300 edges)

| File | Type | Nodes | Edges | Description |
|------|------|-------|-------|-------------|
| `graph_small_er.csv` | Erdos-Renyi | 100 | 250 | Random graph, p=0.05 |
| `graph_small_pl.csv` | Power-Law | 100 | 291 | Preferential attachment, m=3 |
| `graph_small_sw.csv` | Small-World | 100 | 200 | Watts-Strogatz, k=4, p=0.1 |

### Medium Graphs (~1000 nodes, ~5000 edges)

| File | Type | Nodes | Edges | Description |
|------|------|-------|-------|-------------|
| `graph_medium_er.csv` | Erdos-Renyi | 1000 | 5024 | Random graph, p=0.01 |
| `graph_medium_pl.csv` | Power-Law | 1000 | 4975 | Preferential attachment, m=5 |
| `graph_medium_sw.csv` | Small-World | 1000 | 5000 | Watts-Strogatz, k=10, p=0.05 |

### Large Graphs (~5000 nodes, ~50000-60000 edges)

| File | Type | Nodes | Edges | Description |
|------|------|-------|-------|-------------|
| `graph_large_er.csv` | Erdos-Renyi | 5000 | 62456 | Random graph, p=0.005 |
| `graph_large_pl.csv` | Power-Law | 5000 | 49900 | Preferential attachment, m=10 |

### Partitioned Graphs

| Directory | Type | Nodes | Edges | Partitions |
|-----------|------|-------|-------|------------|
| `graph_small_partitioned/` | Erdos-Renyi | 300 | 2182 | 3 |
| `graph_medium_partitioned/` | Power-Law | 1000 | 4971 | 4 |

## Usage Examples

### Extract features from a single graph

```bash
# Small graph
autoconfig graph --input data/graph_small_er.csv --output out/graph_small.yaml

# Medium graph
autoconfig graph --input data/graph_medium_pl.csv --output out/graph_medium.yaml

# Large graph
autoconfig graph --input data/graph_large_er.csv --output out/graph_large.yaml
```

### Extract features from a partitioned graph

```bash
# Small partitioned graph
autoconfig graph --input data/graph_small_partitioned/ --output out/graph_small_part.yaml

# Medium partitioned graph
autoconfig graph --input data/graph_medium_partitioned/ --output out/graph_medium_part.yaml
```

### Complete pipeline

```bash
# Query + Graph + Config + Merge
autoconfig query --input examples/query_gar_match.cu -o out/query.yaml
autoconfig graph --input data/graph_medium_pl.csv -o out/graph.yaml
autoconfig config -n 10 -o out/config.yaml --use-default-catalog
autoconfig merge -q out/query.yaml -g out/graph.yaml -c out/config.yaml -o out/merged.yaml
```

## File Format

All graph files use CSV edge list format:

```csv
src,dst
0,1
0,2
1,2
...
```

- First line is header: `src,dst`
- Each subsequent line is an edge: `source,destination`
- Node IDs can be integers or strings

## Graph Characteristics

### Erdos-Renyi (Random)
- Uniform degree distribution
- Low clustering coefficient
- Small diameter
- Good for testing basic functionality

### Power-Law (Scale-Free)
- Heavy-tailed degree distribution
- High degree skew (few hubs with many connections)
- Low clustering coefficient
- Realistic for social networks, web graphs

### Small-World
- High clustering coefficient
- Short average path length
- Regular lattice with random rewiring
- Models social networks, neural networks

## Regenerate Sample Graphs

To regenerate sample graphs:

```bash
python scripts/generate_sample_graphs.py
```

This will recreate all sample graphs with the same random seed (42) for reproducibility.
