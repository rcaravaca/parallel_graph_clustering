// src/graph.cu
#include "graph.h"
#include <cuda_runtime.h>
#include <iostream>

/**
 * @brief Initializes the graph structure with the given number of nodes.
 *
 * @param graph Reference to the graph structure.
 * @param numNodes Number of nodes in the graph.
 */
void initializeGraph(Graph &graph, int numNodes) {
    graph.numNodes = numNodes;
    graph.adjList = nullptr;
    graph.offset = nullptr;
}

/**
 * @brief Initializes the adjacency list and offset arrays for the graph.
 *
 * @param graph Reference to the graph structure.
 */
void initializeAdjListAndOffsets(Graph &graph) {
    // Example initialization (this would usually be based on actual graph data)
    int adjListSize = graph.numNodes * 10; // Example size, adjust based on actual data
    graph.adjList = new int[adjListSize];
    graph.offset = new int[graph.numNodes];

    // Fill adjList and offset arrays with example data
    // This should be replaced with actual graph data loading
    for (int i = 0; i < adjListSize; i++) {
        graph.adjList[i] = i % graph.numNodes;
    }
    for (int i = 0; i < graph.numNodes; i++) {
        graph.offset[i] = i * 10; // Example offset
    }
}

/**
 * @brief Prints the adjacency list of the graph.
 *
 * @param graph The graph structure containing the adjacency list and offset arrays.
 */
void printGraph(const Graph &graph) {
    std::cout << "Graph adjacency list representation:" << std::endl;
    for (int i = 0; i < graph.numNodes; i++) {
        std::cout << "Node " << i << ":";
        int start = graph.offset[i];
        int end = (i == graph.numNodes - 1) ? graph.numNodes * 10 : graph.offset[i + 1];  // Example range, adjust as necessary
        for (int j = start; j < end; j++) {
            if (graph.adjList[j] != -1) {  // Assuming -1 indicates no edge, adjust based on your data
                std::cout << " " << graph.adjList[j];
            }
        }
        std::cout << std::endl;
    }
}