# Example BFS query code
def BFS(Graph G, vertex source):
    # Initialize
    worklist = [source]
    visited[source] = true
    level[source] = 0
    
    # Frontier-driven iteration
    while !worklist.empty():
        next_worklist = []
        
        for v in worklist:
            # Edge scanning
            for neighbor in G.neighbors(v):
                if !visited[neighbor]:
                    visited[neighbor] = true
                    level[neighbor] = level[v] + 1
                    next_worklist.append(neighbor)
        
        # Update frontier
        worklist = next_worklist
    
    return level
