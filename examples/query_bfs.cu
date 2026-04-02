// BFS (Breadth-First Search) implementation for graph traversal
// Input: Graph G, source vertex
// Output: visited array with BFS order

void BFS(Graph& G, int source) {
    // Initialize visited array
    for (int v = 0; v < numVertices; v++) {
        visited[v] = false;
    }
    
    // BFS worklist
    Worklist worklist;
    worklist.push(source);
    visited[source] = true;
    
    while (!worklist.empty()) {
        // Process current frontier
        for (int v : worklist) {
            // Iterate over neighbors
            for (int neighbor : G.neighbors(v)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    worklist.push(neighbor);
                }
            }
        }
        worklist.swap();
    }
}

// CUDA kernel version
__global__ void BFS_cuda(Graph G, int source) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < G.numVertices) {
        for (int edge = G.rowOffset[v]; edge < G.rowOffset[v + 1]; edge++) {
            int neighbor = G.columnIndices[edge];
            atomicMin(&dist[neighbor], dist[v] + 1);
        }
    }
}
