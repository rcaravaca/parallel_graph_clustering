#include "GraphClustering.h"
#include "kernels/processGraph.h"

void GraphInsertion(Graph& graph, int maxNodes, const std::vector<Digit>& digits) {

    std::cout << "GraphInsertion: Starting the Graph Nodes Insertion..." << std::endl;

    std::vector<int> flatAdjList;
    std::vector<int> adjListSizes;
    std::vector<int> Nodes;
    std::vector<int> flatWeights;
    int numNodes;

    // Flatten the graph into arrays
    graph.flattenGraph(flatAdjList, adjListSizes, Nodes, numNodes);

    // Create flat arrays for the Digit data
    int numDigits = digits.size();
    std::vector<int> rows(numDigits);
    std::vector<int> cols(numDigits);
    std::vector<int> energies(numDigits);

    for (int i = 0; i < numDigits; ++i) {
        rows[i] = digits[i].getRow();
        cols[i] = digits[i].getCol();
        energies[i] = digits[i].getEnergy();
    }

    // Calculate the necessary size for Nodes and adjList
    int NodeSize = maxNodes * 4;       // 3 integers per node + 1 weight
    int adjListSize = maxNodes * 8 * 4;  // Up to 8 neighbors per node, each with 4 integers
    int weightSize = maxNodes * 8;     // 1 weight per neighbor, up to 8 neighbors per node

    Nodes.resize(NodeSize);
    adjListSizes.resize(NodeSize);
    flatAdjList.resize(adjListSize);
    flatWeights.resize(weightSize);

    // Allocate memory on the device
    int *d_adjList, *d_adjListSizes, *d_Nodes, *d_numNodes, *d_rows, *d_cols, *d_energies, *d_flatWeights;
    cudaMalloc(&d_adjList, adjListSize * sizeof(int));  // Allocate enough space for adjList
    cudaMalloc(&d_adjListSizes, maxNodes * sizeof(int));  // One entry per node
    cudaMalloc(&d_Nodes, NodeSize * sizeof(int));  // Allocate enough space for nodes
    cudaMalloc(&d_numNodes, sizeof(int));  // One integer for the number of nodes
    cudaMalloc(&d_rows, rows.size() * sizeof(int));  // Rows of the Digits
    cudaMalloc(&d_cols, cols.size() * sizeof(int));  // Columns of the Digits
    cudaMalloc(&d_energies, energies.size() * sizeof(int));  // Energies of the Digits
    cudaMalloc(&d_flatWeights, weightSize * sizeof(int));  // Weights of the edges

    // Copy data to the device
    cudaMemcpy(d_adjList, flatAdjList.data(), adjListSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjListSizes, adjListSizes.data(), maxNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Nodes, Nodes.data(), NodeSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numNodes, &numNodes, sizeof(int), cudaMemcpyHostToDevice);  // Initialize numNodes to 0 on the device
    cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, energies.data(), energies.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flatWeights, flatWeights.data(), flatWeights.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (numDigits + blockSize - 1) / blockSize;  // Calculate number of blocks

    // Launch the kernel to add nodes
    addNodeToGraphCUDA<<<numBlocks, blockSize>>>(d_adjList, d_adjListSizes, d_Nodes, d_numNodes, maxNodes, d_rows, d_cols, d_energies, numDigits, d_flatWeights);
    // Synchronize the device
    cudaDeviceSynchronize();

    // Copy results back to the host
    cudaMemcpy(flatAdjList.data(), d_adjList, adjListSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(adjListSizes.data(), d_adjListSizes, maxNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Nodes.data(), d_Nodes, NodeSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&numNodes, d_numNodes, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(flatWeights.data(), d_flatWeights, weightSize * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "GraphInsertion: Count of added Nodes: " << numNodes << std::endl;

    // Rebuild the graph on the host
    graph.rebuildGraph(flatAdjList, adjListSizes, Nodes, numNodes);

    // Free memory
    cudaFree(d_adjList);
    cudaFree(d_adjListSizes);
    cudaFree(d_Nodes);
    cudaFree(d_numNodes);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_energies);
    cudaFree(d_flatWeights);
    
    std::cout << "GraphInsertion: Done" << std::endl;

}
