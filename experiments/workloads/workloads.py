"""
Graph workload templates (query codes) for experiments.
These are representative graph algorithms from standard benchmark suites.
"""

# WCC - Weakly Connected Components
# Pattern: Iterative fixpoint computation
query_wcc = """
__global__ void WCC(int* labels, int* edges, int* offsets, int n, bool* changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int my_label = labels[v];
    int min_label = my_label;

    // Vertex scan: iterate over outgoing edges
    int start = offsets[v];
    int end = offsets[v + 1];
    for (int i = start; i < end; i++) {
        int u = edges[i];
        int neighbor_label = labels[u];
        if (neighbor_label < min_label) {
            min_label = neighbor_label;
        }
    }

    // Atomic update for convergence
    if (min_label < my_label) {
        atomicMin(&labels[v], min_label);
        *changed = true;
    }
}
"""

# SSSP - Single-Source Shortest Path
# Pattern: Frontier-driven traversal
query_sssp = """
__global__ void SSSP(int* dist, int* edges, int* offsets, float* weights,
                    int source, int n, bool* active) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    if (!active[v]) return;

    // Edge scan with frontier management
    int start = offsets[v];
    int end = offsets[v + 1];
    float current_dist = dist[v];

    for (int i = start; i < end; i++) {
        int u = edges[i];
        float w = weights[i];
        float new_dist = current_dist + w;

        // Atomic compare-and-swap
        atomicMinFloat(&dist[u], new_dist);
    }

    active[v] = false;
}
"""

# PR - PageRank
# Pattern: Iterative fixpoint computation
query_pr = """
__global__ void pageRank(float* rank, float* new_rank, int* edges,
                         int* offsets, int n, float damping, float* total_outgoing) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    float contribution = 0.0f;

    // Accumulate contributions from neighbors (incoming edges in reverse)
    for (int i = start; i < end; i++) {
        int u = edges[i];
        float outgoing = total_outgoing[u];
        if (outgoing > 0) {
            contribution += rank[u] / outgoing;
        }
    }

    // Dangling node contribution
    new_rank[v] = (1 - damping) / n + damping * contribution;
}
"""

# BFS - Breadth-First Search
# Pattern: Frontier-driven traversal
query_bfs = """
__global__ void BFS(int* depth, int* visited, int* edges, int* offsets,
                    int* frontier, int* new_frontier, int* frontier_size,
                    int n, int level, bool* changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *frontier_size) return;

    int v = frontier[idx];
    int start = offsets[v];
    int end = offsets[v + 1];

    // Frontier expansion
    for (int i = start; i < end; i++) {
        int u = edges[i];
        if (!visited[u]) {
            visited[u] = true;
            depth[u] = level + 1;
            int new_idx = atomicAdd(&new_frontier_size, 1);
            new_frontier[new_idx] = u;
            *changed = true;
        }
    }
}
"""

# SubIso - Subgraph Isomorphism
# Pattern: Recursive pattern matching
query_subiso = """
__global__ void subgraphIsomorphism(int* query_graph, int q_n, int* q_edges,
                                     int* data_graph, int d_n, int* d_edges,
                                     int* mapping, int n_candidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_candidates) return;

    int* current_mapping = mapping + idx * (q_n + 1);
    int matched = recursive_match(query_graph, q_n, q_edges,
                                   data_graph, d_n, d_edges,
                                   current_mapping, 0);

    if (matched) {
        atomicAdd(&count, 1);
    }
}

__device__ bool recursive_match(int* qg, int qn, int* qe,
                                 int* dg, int dn, int* de,
                                 int* mapping, int depth) {
    if (depth == qn) return true;

    // Pattern matching with candidate enumeration
    int v_q = qg[depth];
    for (int v_d = 0; v_d < dn; v_d++) {
        if (is_compatible(v_q, v_d, mapping, depth)) {
            mapping[depth] = v_d;
            if (recursive_match(qg, qn, qe, dg, dn, de, mapping, depth + 1)) {
                return true;
            }
        }
    }
    return false;
}
"""

# GARs - Graph Association Rules
# Pattern: Multi-stage pipeline with rule mining and recursive inference
query_gar_match = """
__global__ void GARMatching(int* antecedent, int ant_n, int* consequent, int cons_n,
                            int* data_graph, int n, int* edges, int* offsets,
                            int* lhs_counts, int* pair_counts, int* result_counts) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    // Stage 1: Antecedent pattern matching (vertex scan)
    bool ant_match = true;
    for (int i = 0; i < ant_n && ant_match; i++) {
        int attr_id = antecedent[i*2];
        int attr_val = antecedent[i*2+1];
        if (data_graph[v*MAX_ATTRS + attr_id] != attr_val) {
            ant_match = false;
        }
    }

    if (!ant_match) return;

    // Stage 2: LHS count aggregation (atomic)
    atomicAdd(&lhs_counts[v], 1);

    // Stage 3: Consequent matching with recursive inference (edge scan)
    int start = offsets[v];
    int end = offsets[v + 1];

    for (int i = start; i < end; i++) {
        int u = edges[i];
        bool cons_match = true;

        for (int j = 0; j < cons_n && cons_match; j++) {
            int attr_id = consequent[j*2];
            int attr_val = consequent[j*2+1];
            if (data_graph[u*MAX_ATTRS + attr_id] != attr_val) {
                cons_match = false;
            }
        }

        if (cons_match) {
            // Pattern pair atomic update
            atomicAdd(&pair_counts[v], 1);
        }
    }

    // Stage 4: Support and confidence calculation
    __syncthreads();
    if (lhs_counts[v] > 0) {
        float support = (float)lhs_counts[v] / n;
        if (pair_counts[v] > 0) {
            float confidence = (float)pair_counts[v] / lhs_counts[v];
            if (confidence >= MIN_CONFIDENCE) {
                atomicAdd(&result_counts[v], 1);
            }
        }
    }
}
"""