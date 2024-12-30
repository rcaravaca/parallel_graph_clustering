#ifndef PROCESS_GRAPH_H
#define PROCESS_GRAPH_H

#include <vector>
#include <iostream>
#include <stdint.h>

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
 * @param flatWeights Flat array for storing the edge weights.
 */
__global__ void addNodeToGraphCUDA(int* adjList, int* adjListSizes, int* nodeIDs, int* numNodes, int maxNodes,
                                   const int* rows, const int* cols, const int* energies, int numDigits, int* flatWeights);

__global__ void addNodeToGraphCUDAv2(int* adjList, int* adjListSizes, int* nodeIDs, int* numNodes, int maxNodes,
                                   const int* rows, const int* cols, const int* energies, int numDigits, int* flatWeights);

__global__ void addNodeToGraphCUDANEventsV1(int* numDigits, int* digitsOffsets, int* adjList, int* adjListSizes, int* Seeds, int* numSeeds, int maxSeeds, const int* rows, const int* cols, const int* energies, int* flatWeights);

__global__ void addNodeToGraphCUDANEventsBase(int* numDigits, int* digitsOffsets, int* adjList, int* adjListSizes, int* Seeds, int* numSeeds, int maxSeeds, const int* rows, const int* cols, const int* energies, int* flatWeights);

__global__ void addNodeToGraphCUDANEventsWithMergedPi0V1(
    int* numDigits,
    int* digitsOffsets,
    int* adjList,
    int* adjListSizes,
    int* Seeds,
    int* numSeeds,
    int maxSeeds,
    const int* rows,
    const int* cols,
    const int* energies,
    int* flatWeights,
    int8_t* isMergedPi0,
    int* numMergedPi0s);

__global__ void addNodeToGraphCUDANEventsWithMergedPi0V2(
    int* numDigits,
    int* digitsOffsets,
    int* adjList,
    int* adjListSizes,
    int* Seeds,
    int* numSeeds,
    int maxSeeds,
    const int* rows,
    const int* cols,
    const int* energies,
    float* flatWeights,
    int* neighborsTotClE,
    int8_t* isMergedPi0,
    int* numMergedPi0s);

// constanst used to identify the position of merged pi0s
enum MergedPi0Positions : int8_t {
    TOP_LEFT = 0,
    TOP,
    TOP_RIGHT,
    LEFT,
    RIGHT,
    BOTTOM_LEFT,
    BOTTOM,
    BOTTOM_RIGHT
};

__global__ void expandPi0sNeighborsV1(
    int* numDigits,
    int* digitsOffsets,
    const int* rows,
    const int* cols,
    const int* energies,
    int* Seeds,
    int maxSeeds,
    int* neighborsTotClE,
    int* numMergedPi0s,
    int* mergedPi0Indexes,
    int8_t* mergedPi0sDirection,
    int* expandedMergedPi0Neighbors,
    int* expandedMergedPi0NeighborsSizes,
    float* expandedMergedPi0Weights);

__global__ void calculateWeightsV1(
    int* d_numSeeds, 
    int* d_Seeds,
    int maxSeeds,
    int* d_adjList,
    int* d_adjListSizes,
    float* d_flatWeights,
    int8_t* d_isMergedPi0,
    int* d_expandedMergedPi0Neighbors,
    int* d_expandedMergedPi0NumNeighbors,
    float* d_expandedMergedPi0Weights,
    int* d_neighborsTotClE
);

#endif // PROCESS_GRAPH_H
