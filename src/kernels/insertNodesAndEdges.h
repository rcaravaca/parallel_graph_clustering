// src/kernels/insertNodesAndEdges.h
#ifndef INSERT_NODES_AND_EDGES_H
#define INSERT_NODES_AND_EDGES_H

#include "../graph.h"

#define IDX2C(i, j, n) ((i) * (n) + (j)) // Macro for accessing the adjacency matrix

static __device__ bool isNodeInserted(int id, int* adjList, int* offset, int numNodes);
static __device__ bool isLocalMaxima(int id, int* energies, int* adjList, int* offset, int numNodes);
static __device__ void addNodeToGraph(int id, int* adjList, int* offset, int numNodes);
static __device__ void addEdge(int id1, int id2, int* adjList, int* offset, int numNodes);
static __device__ bool isMergedCandidate(int id);
static __device__ bool isInMergedPi0(int id, int* mergedPi0);
static __device__ int getFirstSeed(int id, int* adjList, int* offset, int numNodes);
static __device__ int numNeighbors(int id, int* adjList, int* offset, int numNodes);
__global__ void insertNodesAndEdges(int* energies, int* ids, int* adjList, int* offset, int* mergedPi0, int numNodes);

#endif // INSERT_NODES_AND_EDGES_H
