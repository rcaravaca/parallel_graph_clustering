#ifndef PROCESS_GRAPH_H
#define PROCESS_GRAPH_H

#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#define threshold 50

/**
 * @brief CUDA kernel to add nodes and their neighbors with weights to the graph.
 * 
 * This kernel adds new nodes to the adjacency list and records the edge weights
 * between nodes. It processes each node and its neighbors, storing both the 
 * adjacency information and the corresponding weights in flat arrays.
 * 
 * @param adjList Flat adjacency list for storing neighbor nodes.
 * @param adjListSizes Array storing the size of each node's adjacency list.
 * @param nodeIDs Flat array of node information (row, col, energy).
 * @param numNodes Pointer to the number of nodes currently in the graph (updated with atomicAdd).
 * @param maxNodes Maximum number of nodes allowed in the graph.
 * @param rows Array of row indices of the nodes to be added.
 * @param cols Array of column indices of the nodes to be added.
 * @param energies Array of energy values of the nodes to be added.
 * @param numDigits Number of nodes to be processed.
 * @param neighborWeights Array of weights between the nodes and their neighbors.
 * @param flatWeights Flat array for storing the edge weights.
 * @param weightSize Size of the flatWeights array.
 */
__global__ void addNodeToGraphCUDA(int* adjList, int* adjListSizes, int* nodeIDs, int* numNodes, int maxNodes,
                                   const int* rows, const int* cols, const int* energies, int numDigits,
                                   int* flatWeights);


#endif // PROCESS_GRAPH_H
