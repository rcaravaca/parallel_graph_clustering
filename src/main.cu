// src/main.cu
#include <iostream>
#include "graph.h"
#include "utils.h"
#include "kernels/insertNodesAndEdges.h"

int main(int argc, char* argv[]) {
    // Initialize the graph
    Graph graph;

    // Load the graph data (for simplicity, hardcoded in this example)
    int numNodes = 30;  // Example node count
    initializeGraph(graph, numNodes);

    // Load data (for simplicity, hardcoded in this example)
    // In practice, you might load this from the input_data.txt in the data directory
    int* energies = new int[numNodes];
    int* ids = new int[numNodes];
    initializeEnergiesAndIds(energies, ids, numNodes);

    // Allocate memory for the GPU (assuming the adjacency list and offset arrays are already filled)
    cudaMalloc(&(graph.adjList), graph.numNodes * graph.numNodes * sizeof(int));  // Size
    cudaMalloc(&(graph.offset), graph.numNodes * sizeof(int));

    // Initialize the adjacency list and offsets (example initialization)
    initializeAdjListAndOffsets(graph);

    // Print the graph
    printGraph(graph);

    // Allocate additional arrays on the GPU
    int *d_energies, *d_ids, *d_mergedPi0;
    cudaMalloc(&d_energies, numNodes * sizeof(int));
    cudaMalloc(&d_ids, numNodes * sizeof(int));
    cudaMalloc(&d_mergedPi0, numNodes * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_energies, energies, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids, numNodes * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block size
    int blockSize = 256;
    int numBlocks = (numNodes + blockSize - 1) / blockSize;

    // Launch the kernel
    //insertNodesAndEdges<<<numBlocks, blockSize>>>(d_energies, d_ids, graph.adjList, graph.offset, d_mergedPi0, numNodes);

    // Synchronize to ensure the kernel execution is complete
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(d_energies);
    cudaFree(d_ids);
    cudaFree(d_mergedPi0);
    cudaFree(graph.adjList);
    cudaFree(graph.offset);

    // Free host memory
    delete[] energies;
    delete[] ids;

    std::cout << "Parallel Graph Clustering executed successfully." << std::endl;
    return 0;
}
