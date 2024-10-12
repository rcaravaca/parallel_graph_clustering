// src/graph.h
#ifndef GRAPH_H
#define GRAPH_H

struct Graph {
    int *adjList; // Concatenated adjacency list
    int *offset;  // Index of the start of each node in adjList
    int numNodes; // Number of nodes in the graph
};

// Function declarations
void initializeGraph(Graph &graph, int numNodes);
void initializeAdjListAndOffsets(Graph &graph);
void printGraph(const Graph &graph, const int nodes_to_show);
#endif // GRAPH_H
