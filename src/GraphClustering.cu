#include "GraphClustering.h"
#include "kernels/processGraph.h"
#include <thread>
#include <chrono>

void prefixSumPi0sIndexes(int numEvents, int maxSeeds, std::vector<int8_t>& isMergedPi0, std::vector<int>& Pi0Indexes) {
    for (int event = 0; event < numEvents; ++event) {
        int counter = 0;
        for (int seed = 0; seed < maxSeeds; ++seed) {
            if (isMergedPi0[event * maxSeeds + seed] > -1) {
                Pi0Indexes[event * maxSeeds + counter] = seed;
                counter++;
            }
        }
    }
}

void GraphInsertion(Graph& graph, int maxNodes, const std::vector<Digit>& digits) {

    std::cout << "GraphInsertion: Starting the Graph Nodes Insertion..." << std::endl;

    std::vector<int> flatAdjList;
    std::vector<int> adjListSizes;
    std::vector<int> Nodes;
    std::vector<int> flatWeights;
    int numNodes;

    // Flatten the graph into arrays
    graph.flattenGraph(flatAdjList, adjListSizes, Nodes, numNodes); // TODO: What is this for? This is not doing anything atm

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
    int NodeSize = maxNodes * 3;       // 3 integers per node (row, col, energy)
    int adjListSize = maxNodes * 8 * 3;  // Up to 8 neighbors per node, each with 3 integers (row, col, energy)
    int weightSize = maxNodes * 8; // Up to 8 neighbors per node, each with 1 value (weight of the edge)

    Nodes.resize(NodeSize);
    adjListSizes.resize(maxNodes);
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the kernel to add nodes
    cudaEventRecord(start);
    addNodeToGraphCUDA<<<numBlocks, blockSize>>>(d_adjList, d_adjListSizes, d_Nodes, d_numNodes, maxNodes, d_rows, d_cols, d_energies, numDigits, d_flatWeights);
    cudaEventRecord(stop);
    
    // Synchronize the device
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time: " << milliseconds << " ms" << std::endl;

    // Copy results back to the host
    cudaMemcpy(flatAdjList.data(), d_adjList, adjListSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(adjListSizes.data(), d_adjListSizes, maxNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Nodes.data(), d_Nodes, NodeSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&numNodes, d_numNodes, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(flatWeights.data(), d_flatWeights, weightSize * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "GraphInsertion: Count of added Nodes: " << numNodes << std::endl;

    // Rebuild the graph on the host
    graph.rebuildGraph(flatAdjList, adjListSizes, flatWeights, Nodes, numNodes);

    // Free memory
    cudaFree(d_adjList);
    cudaFree(d_adjListSizes);
    cudaFree(d_Nodes);
    cudaFree(d_numNodes);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_energies);
    cudaFree(d_flatWeights);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "GraphInsertion: Done" << std::endl;

}

void GraphInsertionV2(Graph& graph, int maxNodes, const std::vector<Digit>& digits) {

    std::cout << "GraphInsertionV2: Starting the Graph Nodes Insertion..." << std::endl;

    std::vector<int> flatAdjList;
    std::vector<int> adjListSizes;
    std::vector<int> Seeds;
    std::vector<int> flatWeights;
    int numSeeds;

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
    int SeedSize = maxNodes * 3;       // 3 integers per node (row, col, energy)
    int adjListSize = maxNodes * 8 * 3;  // Up to 8 neighbors per node, each with 3 integers (row, col, energy)
    int weightSize = maxNodes * 8; // Up to 8 neighbors per node, each with 1 value (weight of the edge)

    Seeds.resize(SeedSize);
    adjListSizes.resize(maxNodes);
    flatAdjList.resize(adjListSize);
    flatWeights.resize(weightSize);

    // Allocate memory on the device
    int *d_adjList, *d_adjListSizes, *d_Seeds, *d_numSeeds, *d_rows, *d_cols, *d_energies, *d_flatWeights;
    cudaMalloc(&d_adjList, adjListSize * sizeof(int));  // Allocate enough space for adjList
    cudaMalloc(&d_adjListSizes, maxNodes * sizeof(int));  // One entry per node
    cudaMalloc(&d_Seeds, SeedSize * sizeof(int));  // Allocate enough space for nodes
    cudaMalloc(&d_numSeeds, sizeof(int));  // One integer for the number of nodes
    cudaMalloc(&d_rows, rows.size() * sizeof(int));  // Rows of the Digits
    cudaMalloc(&d_cols, cols.size() * sizeof(int));  // Columns of the Digits
    cudaMalloc(&d_energies, energies.size() * sizeof(int));  // Energies of the Digits
    cudaMalloc(&d_flatWeights, weightSize * sizeof(int));  // Weights of the edges

    numSeeds = 0;  // Initialize the number of seeds to 0

    // Copy data to the device
    cudaMemcpy(d_adjList, flatAdjList.data(), adjListSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjListSizes, adjListSizes.data(), maxNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Seeds, Seeds.data(), SeedSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numSeeds, &numSeeds, sizeof(int), cudaMemcpyHostToDevice);  // Initialize numSeeds to 0 on the device
    cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, energies.data(), energies.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flatWeights, flatWeights.data(), flatWeights.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 blockSize = dim3(8, 32); // will use 8 threads for each digit
    int numBlocks = (numDigits + 32 - 1) / 32;  // Calculate number of blocks

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Launch the kernel to add nodes
    addNodeToGraphCUDAv2<<<numBlocks, blockSize>>>(d_adjList, d_adjListSizes, d_Seeds, d_numSeeds, maxNodes, d_rows, d_cols, d_energies, numDigits, d_flatWeights);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time: " << milliseconds << " ms" << std::endl;

    // Synchronize the device

    // Copy results back to the host
    cudaMemcpy(flatAdjList.data(), d_adjList, adjListSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(adjListSizes.data(), d_adjListSizes, maxNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Seeds.data(), d_Seeds, SeedSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&numSeeds, d_numSeeds, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(flatWeights.data(), d_flatWeights, weightSize * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "GraphInsertion: Count of added Seeds: " << numSeeds << std::endl;

    // Rebuild the graph on the host
    graph.rebuildGraph(flatAdjList, adjListSizes, flatWeights, Seeds, numSeeds);

    // Free memory
    cudaFree(d_adjList);
    cudaFree(d_adjListSizes);
    cudaFree(d_Seeds);
    cudaFree(d_numSeeds);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_energies);
    cudaFree(d_flatWeights);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "GraphInsertion: Done" << std::endl;

}

void GraphInsertionNEventsV1(
    std::vector<Graph>& graphs,
    const int maxSeeds,
    const std::vector<Digit>& digits,
    const std::vector<int>& digitsOffsets,
    const std::vector<int>& numDigits) {

    std::cout << "GraphInsertionNEventsV1: Starting the Graph Nodes Insertion..." << std::endl;

    // Assume number of events == number of graphs == size of digitsOffsets == size of numDigits

    if (graphs.size() != digitsOffsets.size() || graphs.size() != numDigits.size()) {
        std::cerr << "GraphInsertionNEventsV1: Error: Number of graphs, digitsOffsets, and numDigits do not match." << std::endl;
        return;
    }

    int numEvents = graphs.size();

    auto initialization = std::chrono::high_resolution_clock::now();

    // info from all graphs in the same vectors
    std::vector<int> flatAdjList;
    std::vector<int> adjListSizes;
    std::vector<int> Seeds;
    std::vector<int> flatWeights;

    std::vector<int> numSeeds; // Number of seeds in each graph
    
    int totalNumberOfDigits = digits.size();
    std::vector<int> rows(totalNumberOfDigits);
    std::vector<int> cols(totalNumberOfDigits);
    std::vector<int> energies(totalNumberOfDigits);

    for (int i = 0; i < totalNumberOfDigits; ++i) {
        rows[i] = digits[i].getRow();
        cols[i] = digits[i].getCol();
        energies[i] = digits[i].getEnergy();
    }

    int SeedSize = numEvents * maxSeeds * 3; // 3 integers per node (row, col, energy)
    int adjListSize = numEvents * maxSeeds * 8 * 3; // Up to 8 neighbors per node, each with 3 integers (row, col, energy)
    int weightSize = numEvents * maxSeeds * 8; // Up to 8 neighbors per node, each with 1 value (weight of the edge)

    Seeds.resize(SeedSize);
    adjListSizes.resize(numEvents * maxSeeds);
    flatAdjList.resize(adjListSize);
    flatWeights.resize(weightSize);
    numSeeds.resize(numEvents);

    // Allocate memory on the device
    int *d_numDigits, *d_digitsOffsets, *d_adjList, *d_adjListSizes, *d_Seeds, *d_numSeeds, *d_rows, *d_cols, *d_energies, *d_flatWeights;
    cudaMalloc(&d_numDigits, numEvents * sizeof(int));
    cudaMalloc(&d_digitsOffsets, numEvents * sizeof(int));
    cudaMalloc(&d_adjList, adjListSize * sizeof(int));
    cudaMalloc(&d_adjListSizes, numEvents * maxSeeds * sizeof(int));
    cudaMalloc(&d_Seeds, SeedSize * sizeof(int));
    cudaMalloc(&d_numSeeds, numEvents * sizeof(int));
    cudaMalloc(&d_rows, rows.size() * sizeof(int));
    cudaMalloc(&d_cols, cols.size() * sizeof(int));
    cudaMalloc(&d_energies, energies.size() * sizeof(int));
    cudaMalloc(&d_flatWeights, weightSize * sizeof(int));

    // int totalBytesReserved = numEvents * sizeof(int) + numEvents * sizeof(int) + numEvents * sizeof(int) + adjListSize * sizeof(int) + numEvents * maxSeeds * sizeof(int) + SeedSize * sizeof(int) + numEvents * sizeof(int) + rows.size() * sizeof(int) + cols.size() * sizeof(int) + energies.size() * sizeof(int) + weightSize * sizeof(int);

    // std::cout << "GraphInsertionNEventsV1: Total bytes reserved: " << totalBytesReserved << std::endl;

    cudaMemcpy(d_numDigits, numDigits.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice); // TODO: this could be created in a coalesced way, although its just a value per block
    cudaMemcpy(d_digitsOffsets, digitsOffsets.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjList, flatAdjList.data(), adjListSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjListSizes, adjListSizes.data(), numEvents * maxSeeds * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Seeds, Seeds.data(), SeedSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numSeeds, numSeeds.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, energies.data(), energies.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flatWeights, flatWeights.data(), weightSize * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 blockSize = dim3(8, 32); // will use 8 threads for each digit

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    auto endinitialization = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedinitialization = endinitialization - initialization;
    std::cout << "Elapsed initilization on GPU: " << elapsedinitialization.count() << " seconds" << std::endl;

    addNodeToGraphCUDANEventsV1<<<numEvents, blockSize>>>(d_numDigits, d_digitsOffsets, d_adjList, d_adjListSizes, d_Seeds, d_numSeeds, maxSeeds, d_rows, d_cols, d_energies, d_flatWeights);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time kernel addNodeToGraphCUDANEventsV1: " << milliseconds << " ms" << std::endl;

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "GraphInsertionNEventsV1: CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    // Copy results back to the host
    cudaMemcpy(flatAdjList.data(), d_adjList, adjListSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(adjListSizes.data(), d_adjListSizes, numEvents * maxSeeds * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Seeds.data(), d_Seeds, SeedSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numSeeds.data(), d_numSeeds, numEvents * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(flatWeights.data(), d_flatWeights, weightSize * sizeof(int), cudaMemcpyDeviceToHost);

    auto startcpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numEvents; ++i) { 
        std::vector<int> subAdjList(flatAdjList.begin() + i * maxSeeds * 8 * 3, flatAdjList.begin() + (i+1) * maxSeeds * 8 * 3);
        std::vector<int> subAdjListSizes(adjListSizes.begin() + i * maxSeeds, adjListSizes.begin() + (i+1) * maxSeeds);
        std::vector<int> subFlatWeights(flatWeights.begin() + i * maxSeeds * 8, flatWeights.begin() + (i+1) * maxSeeds * 8);
        std::vector<int> subSeeds(Seeds.begin() + i * maxSeeds * 3, Seeds.begin() + (i+1) * maxSeeds * 3);

        graphs[i].rebuildGraph(subAdjList, subAdjListSizes, subFlatWeights, subSeeds, numSeeds[i]);
        // graphs[i].GraphSummary();
    }

    auto endcpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endcpu - startcpu;
    std::cout << "Elapsed rebuilding on GPU: " << elapsed.count() << " seconds" << std::endl;

    // Free memory
    cudaFree(d_numDigits);
    cudaFree(d_digitsOffsets);
    cudaFree(d_adjList);
    cudaFree(d_adjListSizes);
    cudaFree(d_Seeds);
    cudaFree(d_numSeeds);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_energies);
    cudaFree(d_flatWeights);
}

// this is the very inefficient version
void GraphInsertionNEventsBase(
    std::vector<Graph>& graphs,
    const int maxSeeds,
    const std::vector<Digit>& digits,
    const std::vector<int>& digitsOffsets,
    const std::vector<int>& numDigits) {

    std::cout << "GraphInsertionNEventsBase: Starting the Graph Nodes Insertion..." << std::endl;

    // Assume number of events == number of graphs == size of digitsOffsets == size of numDigits

    if (graphs.size() != digitsOffsets.size() || graphs.size() != numDigits.size()) {
        std::cerr << "GraphInsertionNEventsBase: Error: Number of graphs, digitsOffsets, and numDigits do not match." << std::endl;
        return;
    }

    int numEvents = graphs.size();

    auto initialization = std::chrono::high_resolution_clock::now();

    // info from all graphs in the same vectors
    std::vector<int> flatAdjList;
    std::vector<int> adjListSizes;
    std::vector<int> Seeds;
    std::vector<int> flatWeights;

    std::vector<int> numSeeds; // Number of seeds in each graph
    
    int totalNumberOfDigits = digits.size();
    std::vector<int> rows(totalNumberOfDigits);
    std::vector<int> cols(totalNumberOfDigits);
    std::vector<int> energies(totalNumberOfDigits);

    for (int i = 0; i < totalNumberOfDigits; ++i) {
        rows[i] = digits[i].getRow();
        cols[i] = digits[i].getCol();
        energies[i] = digits[i].getEnergy();
    }

    int SeedSize = numEvents * maxSeeds * 3; // 3 integers per node (row, col, energy)
    int adjListSize = numEvents * maxSeeds * 8 * 3; // Up to 8 neighbors per node, each with 3 integers (row, col, energy)
    int weightSize = numEvents * maxSeeds * 8; // Up to 8 neighbors per node, each with 1 value (weight of the edge)

    Seeds.resize(SeedSize);
    adjListSizes.resize(numEvents * maxSeeds);
    flatAdjList.resize(adjListSize);
    flatWeights.resize(weightSize);
    numSeeds.resize(numEvents);

    // Allocate memory on the device
    int *d_numDigits, *d_digitsOffsets, *d_adjList, *d_adjListSizes, *d_Seeds, *d_numSeeds, *d_rows, *d_cols, *d_energies, *d_flatWeights;
    cudaMalloc(&d_numDigits, numEvents * sizeof(int));
    cudaMalloc(&d_digitsOffsets, numEvents * sizeof(int));
    cudaMalloc(&d_adjList, adjListSize * sizeof(int));
    cudaMalloc(&d_adjListSizes, numEvents * maxSeeds * sizeof(int));
    cudaMalloc(&d_Seeds, SeedSize * sizeof(int));
    cudaMalloc(&d_numSeeds, numEvents * sizeof(int));
    cudaMalloc(&d_rows, rows.size() * sizeof(int));
    cudaMalloc(&d_cols, cols.size() * sizeof(int));
    cudaMalloc(&d_energies, energies.size() * sizeof(int));
    cudaMalloc(&d_flatWeights, weightSize * sizeof(int));

    // int totalBytesReserved = numEvents * sizeof(int) + numEvents * sizeof(int) + numEvents * sizeof(int) + adjListSize * sizeof(int) + numEvents * maxSeeds * sizeof(int) + SeedSize * sizeof(int) + numEvents * sizeof(int) + rows.size() * sizeof(int) + cols.size() * sizeof(int) + energies.size() * sizeof(int) + weightSize * sizeof(int);

    // std::cout << "GraphInsertionNEventsBase: Total bytes reserved: " << totalBytesReserved << std::endl;

    cudaMemcpy(d_numDigits, numDigits.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice); // TODO: this could be created in a coalesced way, although its just a value per block
    cudaMemcpy(d_digitsOffsets, digitsOffsets.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjList, flatAdjList.data(), adjListSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjListSizes, adjListSizes.data(), numEvents * maxSeeds * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Seeds, Seeds.data(), SeedSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numSeeds, numSeeds.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, energies.data(), energies.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flatWeights, flatWeights.data(), weightSize * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 blockSize = dim3(256); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    auto endinitialization = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedinitialization = endinitialization - initialization;
    std::cout << "Elapsed initilization on GPU: " << elapsedinitialization.count() << " seconds" << std::endl;

    addNodeToGraphCUDANEventsBase<<<numEvents, blockSize>>>(d_numDigits, d_digitsOffsets, d_adjList, d_adjListSizes, d_Seeds, d_numSeeds, maxSeeds, d_rows, d_cols, d_energies, d_flatWeights);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time kernel addNodeToGraphCUDANEventsBase: " << milliseconds << " ms" << std::endl;

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "GraphInsertionNEventsBase: CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    // Copy results back to the host
    cudaMemcpy(flatAdjList.data(), d_adjList, adjListSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(adjListSizes.data(), d_adjListSizes, numEvents * maxSeeds * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Seeds.data(), d_Seeds, SeedSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numSeeds.data(), d_numSeeds, numEvents * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(flatWeights.data(), d_flatWeights, weightSize * sizeof(int), cudaMemcpyDeviceToHost);

    auto startcpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numEvents; ++i) { 
        std::vector<int> subAdjList(flatAdjList.begin() + i * maxSeeds * 8 * 3, flatAdjList.begin() + (i+1) * maxSeeds * 8 * 3);
        std::vector<int> subAdjListSizes(adjListSizes.begin() + i * maxSeeds, adjListSizes.begin() + (i+1) * maxSeeds);
        std::vector<int> subFlatWeights(flatWeights.begin() + i * maxSeeds * 8, flatWeights.begin() + (i+1) * maxSeeds * 8);
        std::vector<int> subSeeds(Seeds.begin() + i * maxSeeds * 3, Seeds.begin() + (i+1) * maxSeeds * 3);

        graphs[i].rebuildGraph(subAdjList, subAdjListSizes, subFlatWeights, subSeeds, numSeeds[i]);
        // graphs[i].GraphSummary();
    }

    auto endcpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endcpu - startcpu;
    std::cout << "Elapsed rebuilding on GPU: " << elapsed.count() << " seconds" << std::endl;

    // Free memory
    cudaFree(d_numDigits);
    cudaFree(d_digitsOffsets);
    cudaFree(d_adjList);
    cudaFree(d_adjListSizes);
    cudaFree(d_Seeds);
    cudaFree(d_numSeeds);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_energies);
    cudaFree(d_flatWeights);
}

void GraphInsertionNEventsWithPi0V1(
    std::vector<Graph>& graphs,
    int maxSeeds,
    const std::vector<Digit>& digits,
    const std::vector<int>& digitsOffsets,
    const std::vector<int>& numDigits) {

    std::cout << "GraphInsertionNEventsWithPi0V1: Starting the Graph Nodes Insertion..." << std::endl;

    // Assume number of events == number of graphs == size of digitsOffsets == size of numDigits

    if (graphs.size() != digitsOffsets.size() || graphs.size() != numDigits.size()) {
        std::cerr << "GraphInsertionNEventsWithPi0V1: Error: Number of graphs, digitsOffsets, and numDigits do not match." << std::endl;
        return;
    }

    int numEvents = graphs.size();

    // info from all graphs in the same vectors
    std::vector<int> flatAdjList;
    std::vector<int> adjListSizes;
    std::vector<int> Seeds;
    std::vector<int> flatWeights;
    
    std::vector<int> numSeeds; // Number of seeds in each graph

    std::vector<int8_t> isMergedPi0; // will contain 0 if not, and a constant indicating the direction of the merge if it is
    std::vector<int> mergedPi0AdjListSizes;
    std::vector<int> expandedMergedPi0Neighbors;
    std::vector<int> expandedMergedPi0Weights;

    std::vector<int> numMergedPi0; // Number of merged pi0s in each graph
    
    int totalNumberOfDigits = digits.size();
    std::vector<int> rows(totalNumberOfDigits);
    std::vector<int> cols(totalNumberOfDigits);
    std::vector<int> energies(totalNumberOfDigits);

    for (int i = 0; i < totalNumberOfDigits; ++i) {
        rows[i] = digits[i].getRow();
        cols[i] = digits[i].getCol();
        energies[i] = digits[i].getEnergy();
    }

    maxSeeds = maxSeeds % 9 == 0 ? maxSeeds / 9 : maxSeeds / 9 + 1;

    int SeedSize = numEvents * maxSeeds * 3; // 3 integers per node (row, col, energy)
    int adjListSize = numEvents * maxSeeds * 8 * 3; // Up to 8 neighbors per node, each with 3 integers (row, col, energy)
    int weightSize = numEvents * maxSeeds * 8; // Up to 8 neighbors per node, each with 1 value (weight of the edge)

    Seeds.resize(SeedSize);
    adjListSizes.resize(numEvents * maxSeeds);
    flatAdjList.resize(adjListSize);
    flatWeights.resize(weightSize);
    numSeeds.resize(numEvents);

    int expandedMergedPi0AdjListSize = numEvents * maxSeeds * 5 * 3; // Up to 5 additional neighbors per merged pi0 (diagonal cases), 
                                                                     // each with 3 integers (row, col, energy)
    int expandedMergedPi0WeightsSize = numEvents * maxSeeds * 5; 
    isMergedPi0.resize(numEvents * maxSeeds, -1);
    mergedPi0AdjListSizes.resize(numEvents * maxSeeds);
    expandedMergedPi0Neighbors.resize(expandedMergedPi0AdjListSize);
    expandedMergedPi0Weights.resize(expandedMergedPi0WeightsSize);
    numMergedPi0.resize(numEvents);

    // Allocate memory on the device
    int *d_numDigits, *d_digitsOffsets, *d_adjList, *d_adjListSizes, *d_Seeds, *d_numSeeds, *d_rows, *d_cols, *d_energies, *d_flatWeights,
            *d_mergedPi0AdjListSizes, *d_expandedMergedPi0Neighbors, *d_expandedMergedPi0Weights, *d_numMergedPi0;
    int8_t *d_isMergedPi0;
    cudaMalloc(&d_numDigits, numEvents * sizeof(int));
    cudaMalloc(&d_digitsOffsets, numEvents * sizeof(int));
    cudaMalloc(&d_adjList, adjListSize * sizeof(int));
    cudaMalloc(&d_adjListSizes, numEvents * maxSeeds * sizeof(int));
    cudaMalloc(&d_Seeds, SeedSize * sizeof(int));
    cudaMalloc(&d_numSeeds, numEvents * sizeof(int));
    cudaMalloc(&d_rows, rows.size() * sizeof(int));
    cudaMalloc(&d_cols, cols.size() * sizeof(int));
    cudaMalloc(&d_energies, energies.size() * sizeof(int));
    cudaMalloc(&d_flatWeights, weightSize * sizeof(int));
    cudaMalloc(&d_isMergedPi0, numEvents * maxSeeds * sizeof(int8_t));
    cudaMalloc(&d_mergedPi0AdjListSizes, numEvents * maxSeeds * sizeof(int));
    cudaMalloc(&d_expandedMergedPi0Neighbors, expandedMergedPi0AdjListSize * sizeof(int));
    cudaMalloc(&d_expandedMergedPi0Weights, expandedMergedPi0WeightsSize * sizeof(int));
    cudaMalloc(&d_numMergedPi0, numEvents * sizeof(int));

    // int totalBytesReserved = numEvents * sizeof(int) + numEvents * sizeof(int) + numEvents * sizeof(int) + adjListSize * sizeof(int) + numEvents * maxSeeds * sizeof(int) + SeedSize * sizeof(int) + numEvents * sizeof(int) + rows.size() * sizeof(int) + cols.size() * sizeof(int) + energies.size() * sizeof(int) + weightSize * sizeof(int) + numEvents * maxSeeds * sizeof(uint8_t) + numEvents * maxSeeds * sizeof(int) + expandedMergedPi0AdjListSize * sizeof(int) + expandedMergedPi0WeightsSize * sizeof(int) + numEvents * sizeof(int);

    // std::cout << "GraphInsertionNEventsWithPi0V1: Total bytes reserved: " << totalBytesReserved << std::endl;
    // return;

    cudaMemcpy(d_numDigits, numDigits.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice); // TODO: this could be created in a coalesced way, although its just a value per block
    cudaMemcpy(d_digitsOffsets, digitsOffsets.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjList, flatAdjList.data(), adjListSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjListSizes, adjListSizes.data(), numEvents * maxSeeds * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Seeds, Seeds.data(), SeedSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numSeeds, numSeeds.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, energies.data(), energies.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flatWeights, flatWeights.data(), weightSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_isMergedPi0, isMergedPi0.data(), numEvents * maxSeeds * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mergedPi0AdjListSizes, mergedPi0AdjListSizes.data(), numEvents * maxSeeds * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expandedMergedPi0Neighbors, expandedMergedPi0Neighbors.data(), expandedMergedPi0AdjListSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expandedMergedPi0Weights, expandedMergedPi0Weights.data(), expandedMergedPi0WeightsSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numMergedPi0, numMergedPi0.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 blockSize = dim3(8, 32); // will use 8 threads for each digit

    addNodeToGraphCUDANEventsWithMergedPi0V1<<<numEvents, blockSize>>>(d_numDigits, d_digitsOffsets, d_adjList, d_adjListSizes, d_Seeds, d_numSeeds, maxSeeds, d_rows, d_cols, d_energies, d_flatWeights, d_isMergedPi0, d_numMergedPi0);
    cudaMemcpy(isMergedPi0.data(), d_isMergedPi0, numEvents * maxSeeds * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(numMergedPi0.data(), d_numMergedPi0, numEvents * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    std::vector<int> Pi0Indexes(numEvents * maxSeeds);
    prefixSumPi0sIndexes(numEvents, maxSeeds, isMergedPi0, Pi0Indexes); // entonces no uso el de numPi0s?
    
    for (int i = 0; i < numMergedPi0[0]; ++i) {
        std::cout << "Merged Pi0: " << i << " at index: " << Pi0Indexes[i] << " with direction: " << int(isMergedPi0[Pi0Indexes[i]]) << std::endl;
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "GraphInsertionNEventsWithPi0V1: CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    int *d_Pi0Indexes;
    cudaMalloc(&d_Pi0Indexes, numEvents * maxSeeds * sizeof(int));
    cudaMemcpy(d_Pi0Indexes, Pi0Indexes.data(), numEvents * maxSeeds * sizeof(int), cudaMemcpyHostToDevice);

    // expandPi0sNeighborsV1<<<1, dim3(8,1)>>>(d_numDigits, d_digitsOffsets, d_rows, d_cols, d_energies, d_Seeds, maxSeeds, d_numMergedPi0, d_Pi0Indexes, d_isMergedPi0);

    // Copy results back to the host
    cudaMemcpy(flatAdjList.data(), d_adjList, adjListSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(adjListSizes.data(), d_adjListSizes, numEvents * maxSeeds * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Seeds.data(), d_Seeds, SeedSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numSeeds.data(), d_numSeeds, numEvents * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(flatWeights.data(), d_flatWeights, weightSize * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numEvents; ++i) { 
        std::vector<int> subAdjList(flatAdjList.begin() + i * maxSeeds * 8 * 3, flatAdjList.begin() + (i+1) * maxSeeds * 8 * 3);
        std::vector<int> subAdjListSizes(adjListSizes.begin() + i * maxSeeds, adjListSizes.begin() + (i+1) * maxSeeds);
        std::vector<int> subFlatWeights(flatWeights.begin() + i * maxSeeds * 8, flatWeights.begin() + (i+1) * maxSeeds * 8);
        std::vector<int> subSeeds(Seeds.begin() + i * maxSeeds * 3, Seeds.begin() + (i+1) * maxSeeds * 3);

        // graphs[i].rebuildGraph(subAdjList, subAdjListSizes, subFlatWeights, subSeeds, numSeeds[i]);
        // graphs[i].GraphSummary();
    }

    // Free memory
    cudaFree(d_numDigits);
    cudaFree(d_digitsOffsets);
    cudaFree(d_adjList);
    cudaFree(d_adjListSizes);
    cudaFree(d_Seeds);
    cudaFree(d_numSeeds);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_energies);
    cudaFree(d_flatWeights);
}

void GraphInsertionNEventsWithPi0V2(
    std::vector<Graph>& graphs,
    int maxSeeds,
    const std::vector<Digit>& digits,
    const std::vector<int>& digitsOffsets,
    const std::vector<int>& numDigits) {

    std::cout << "GraphInsertionNEventsWithPi0V2: Starting the Graph Nodes Insertion..." << std::endl;

    // Assume number of events == number of graphs == size of digitsOffsets == size of numDigits

    if (graphs.size() != digitsOffsets.size() || graphs.size() != numDigits.size()) {
        std::cerr << "GraphInsertionNEventsWithPi0V2: Error: Number of graphs, digitsOffsets, and numDigits do not match." << std::endl;
        return;
    }

    int numEvents = graphs.size();

    // info from all graphs in the same vectors
    std::vector<int> flatAdjList;
    std::vector<int> adjListSizes;
    std::vector<int> Seeds;
    std::vector<float> flatWeights;
    std::vector<int> neighborsTotClE;
    
    std::vector<int> numSeeds; // Number of seeds in each graph

    std::vector<int8_t> isMergedPi0; // will contain 0 if not, and a constant indicating the direction of the merge if it is
    std::vector<int> mergedPi0AdjListSizes;
    std::vector<int> expandedMergedPi0Neighbors;
    std::vector<int> expandedMergedPi0NumNeighbors;
    std::vector<float> expandedMergedPi0Weights;

    std::vector<int> numMergedPi0; // Number of merged pi0s in each graph
    
    int totalNumberOfDigits = digits.size();
    std::vector<int> rows(totalNumberOfDigits);
    std::vector<int> cols(totalNumberOfDigits);
    std::vector<int> energies(totalNumberOfDigits);

    for (int i = 0; i < totalNumberOfDigits; ++i) {
        rows[i] = digits[i].getRow();
        cols[i] = digits[i].getCol();
        energies[i] = digits[i].getEnergy();
    }

    maxSeeds = maxSeeds % 9 == 0 ? maxSeeds / 9 : maxSeeds / 9 + 1;

    int SeedSize = numEvents * maxSeeds * 3; // 3 integers per node (row, col, energy)
    int adjListSize = numEvents * maxSeeds * 8 * 3; // Up to 8 neighbors per node, each with 3 integers (row, col, energy)
    int weightSize = numEvents * maxSeeds * 8; // Up to 8 neighbors per node, each with 1 value (weight of the edge)

    Seeds.resize(SeedSize);
    adjListSizes.resize(numEvents * maxSeeds);
    flatAdjList.resize(adjListSize);
    flatWeights.resize(weightSize);
    neighborsTotClE.resize(numEvents * 64 * 58);
    numSeeds.resize(numEvents);

    int expandedMergedPi0AdjListSize = numEvents * maxSeeds * 5 * 3; // Up to 5 additional neighbors per merged pi0 (diagonal cases), 
                                                                     // each with 3 integers (row, col, energy)
    int expandedMergedPi0WeightsSize = numEvents * maxSeeds * 5; 
    isMergedPi0.resize(numEvents * maxSeeds, -1);
    mergedPi0AdjListSizes.resize(numEvents * maxSeeds);
    expandedMergedPi0Neighbors.resize(expandedMergedPi0AdjListSize);
    expandedMergedPi0NumNeighbors.resize(numEvents * maxSeeds);
    expandedMergedPi0Weights.resize(expandedMergedPi0WeightsSize);
    numMergedPi0.resize(numEvents);

    // Allocate memory on the device
    int *d_numDigits, *d_digitsOffsets, *d_adjList, *d_adjListSizes, *d_Seeds, *d_numSeeds, *d_rows, *d_cols, *d_energies,
            *d_neighborsTotClE, *d_expandedMergedPi0Neighbors, *d_expandedMergedPi0NumNeighbors, *d_numMergedPi0;
    float *d_flatWeights, *d_expandedMergedPi0Weights;
    int8_t *d_isMergedPi0;
    cudaMalloc(&d_numDigits, numEvents * sizeof(int));
    cudaMalloc(&d_digitsOffsets, numEvents * sizeof(int));
    cudaMalloc(&d_adjList, adjListSize * sizeof(int));
    cudaMalloc(&d_adjListSizes, numEvents * maxSeeds * sizeof(int));
    cudaMalloc(&d_Seeds, SeedSize * sizeof(int));
    cudaMalloc(&d_numSeeds, numEvents * sizeof(int));
    cudaMalloc(&d_rows, rows.size() * sizeof(int));
    cudaMalloc(&d_cols, cols.size() * sizeof(int));
    cudaMalloc(&d_energies, energies.size() * sizeof(int));
    cudaMalloc(&d_flatWeights, weightSize * sizeof(float));
    cudaMalloc(&d_neighborsTotClE, numEvents * 64 * 58 * sizeof(int));
    cudaMalloc(&d_isMergedPi0, numEvents * maxSeeds * sizeof(int8_t));
    cudaMalloc(&d_expandedMergedPi0Neighbors, expandedMergedPi0AdjListSize * sizeof(int));
    cudaMalloc(&d_expandedMergedPi0NumNeighbors, numEvents * maxSeeds * sizeof(int));
    cudaMalloc(&d_expandedMergedPi0Weights, expandedMergedPi0WeightsSize * sizeof(float));
    cudaMalloc(&d_numMergedPi0, numEvents * sizeof(int));

    // int totalBytesReserved = numEvents * sizeof(int) + numEvents * sizeof(int) + numEvents * sizeof(int) + adjListSize * sizeof(int) + numEvents * maxSeeds * sizeof(int) + SeedSize * sizeof(int) + numEvents * sizeof(int) + rows.size() * sizeof(int) + cols.size() * sizeof(int) + energies.size() * sizeof(int) + weightSize * sizeof(int) + numEvents * maxSeeds * sizeof(uint8_t) + numEvents * maxSeeds * sizeof(int) + expandedMergedPi0AdjListSize * sizeof(int) + expandedMergedPi0WeightsSize * sizeof(int) + numEvents * sizeof(int);

    // std::cout << "GraphInsertionNEventsWithPi0V2: Total bytes reserved: " << totalBytesReserved << std::endl;
    // return;

    cudaMemcpy(d_numDigits, numDigits.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice); // TODO: this could be created in a coalesced way, although its just a value per block
    cudaMemcpy(d_digitsOffsets, digitsOffsets.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjList, flatAdjList.data(), adjListSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjListSizes, adjListSizes.data(), numEvents * maxSeeds * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Seeds, Seeds.data(), SeedSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numSeeds, numSeeds.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, energies.data(), energies.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flatWeights, flatWeights.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighborsTotClE, neighborsTotClE.data(), numEvents * 64 * 58 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_isMergedPi0, isMergedPi0.data(), numEvents * maxSeeds * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expandedMergedPi0Neighbors, expandedMergedPi0Neighbors.data(), expandedMergedPi0AdjListSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expandedMergedPi0NumNeighbors, expandedMergedPi0NumNeighbors.data(), numEvents * maxSeeds * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expandedMergedPi0Weights, expandedMergedPi0Weights.data(), expandedMergedPi0WeightsSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numMergedPi0, numMergedPi0.data(), numEvents * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 blockSize = dim3(8, 32); // will use 8 threads for each digit

    addNodeToGraphCUDANEventsWithMergedPi0V2<<<numEvents, blockSize>>>(d_numDigits, d_digitsOffsets, d_adjList, d_adjListSizes, d_Seeds, d_numSeeds, maxSeeds, d_rows, d_cols, d_energies, d_flatWeights, d_neighborsTotClE, d_isMergedPi0, d_numMergedPi0);
    cudaMemcpy(isMergedPi0.data(), d_isMergedPi0, numEvents * maxSeeds * sizeof(int8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(numMergedPi0.data(), d_numMergedPi0, numEvents * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    std::vector<int> Pi0Indexes(numEvents * maxSeeds);
    prefixSumPi0sIndexes(numEvents, maxSeeds, isMergedPi0, Pi0Indexes); // entonces no uso el de numPi0s?
    
    // for (int i = 0; i < numMergedPi0[0]; ++i) {
    //     std::cout << "Merged Pi0: " << i << " at index: " << Pi0Indexes[i] << " with direction: " << int(isMergedPi0[Pi0Indexes[i]]) << std::endl;
    // }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "GraphInsertionNEventsWithPi0V2: CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    int *d_Pi0Indexes;
    cudaMalloc(&d_Pi0Indexes, numEvents * maxSeeds * sizeof(int));
    cudaMemcpy(d_Pi0Indexes, Pi0Indexes.data(), numEvents * maxSeeds * sizeof(int), cudaMemcpyHostToDevice);

    expandPi0sNeighborsV1<<<numEvents, dim3(8,32)>>>(d_numDigits, d_digitsOffsets, d_rows, d_cols, d_energies, d_Seeds, maxSeeds, d_neighborsTotClE,  d_numMergedPi0, d_Pi0Indexes, d_isMergedPi0, d_expandedMergedPi0Neighbors, d_expandedMergedPi0NumNeighbors, d_expandedMergedPi0Weights);

    // Copy results back to the host
    cudaMemcpy(flatAdjList.data(), d_adjList, adjListSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(adjListSizes.data(), d_adjListSizes, numEvents * maxSeeds * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Seeds.data(), d_Seeds, SeedSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numSeeds.data(), d_numSeeds, numEvents * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(neighborsTotClE.data(), d_neighborsTotClE, numEvents * 64 * 58 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(expandedMergedPi0Neighbors.data(), d_expandedMergedPi0Neighbors, expandedMergedPi0AdjListSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(expandedMergedPi0NumNeighbors.data(), d_expandedMergedPi0NumNeighbors, numEvents * maxSeeds * sizeof(int), cudaMemcpyDeviceToHost);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "expandPi0sNeighborsV1: CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    calculateWeightsV1<<<numEvents, dim3(8,32)>>>(d_numSeeds, d_Seeds, maxSeeds, d_adjList, d_adjListSizes, d_flatWeights, d_isMergedPi0, d_expandedMergedPi0Neighbors, d_expandedMergedPi0NumNeighbors, d_expandedMergedPi0Weights, d_neighborsTotClE);

    cudaMemcpy(flatWeights.data(), d_flatWeights, weightSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(expandedMergedPi0Weights.data(), d_expandedMergedPi0Weights, expandedMergedPi0WeightsSize * sizeof(float), cudaMemcpyDeviceToHost);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "calculateWeightsV1: CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    for (int i = 0; i < numSeeds[0]; i++) {
        std::cout << "Seed: " << i << " is at (" << Seeds[i*3] << ", " << Seeds[i*3+1] << ") with energy " << Seeds[i*3+2] << 
            " and has " << adjListSizes[i] << " neighbors. Is merged pi0: " << int(isMergedPi0[i]) << ". Expanded neighbors: " << expandedMergedPi0NumNeighbors[i] << ". Cluster energy: " << neighborsTotClE[Seeds[i*3] * 64 + Seeds[i*3+1]] << std::endl;
        for (int j = 0; j < adjListSizes[i]; j++) {
            std::cout << "Neighbor " << j << " is at (" << flatAdjList[i*8*3 + j*3] << ", " << flatAdjList[i*8*3 + j*3+1] << ") with energy " << flatAdjList[i*8*3 + j*3+2] << " and weight " << flatWeights[i*8 + j] << ". Total cluster energy: " << neighborsTotClE[flatAdjList[i*8*3 + j*3] * 64 + flatAdjList[i*8*3 + j*3+1] ] << ". Neighbor weight: " << flatWeights[i*8 + j] << std::endl;
        }
        for (int j = 0; j < expandedMergedPi0NumNeighbors[i]; j++) {
            std::cout << "Expanded neighbor " << j << " is at (" << expandedMergedPi0Neighbors[i*5*3 + j*3] << ", " << expandedMergedPi0Neighbors[i*5*3 + j*3+1] << ") with energy " << expandedMergedPi0Neighbors[i*5*3 + j*3+2] << " and weight " << expandedMergedPi0Weights[i*5 + j] << std::endl;
        }
    }

    for (int i = 0; i < numEvents; ++i) { 
        std::vector<int> subAdjList(flatAdjList.begin() + i * maxSeeds * 8 * 3, flatAdjList.begin() + (i+1) * maxSeeds * 8 * 3);
        std::vector<int> subAdjListSizes(adjListSizes.begin() + i * maxSeeds, adjListSizes.begin() + (i+1) * maxSeeds);
        std::vector<int> subFlatWeights(flatWeights.begin() + i * maxSeeds * 8, flatWeights.begin() + (i+1) * maxSeeds * 8);
        std::vector<int> subSeeds(Seeds.begin() + i * maxSeeds * 3, Seeds.begin() + (i+1) * maxSeeds * 3);

        // graphs[i].rebuildGraph(subAdjList, subAdjListSizes, subFlatWeights, subSeeds, numSeeds[i]);
        // graphs[i].GraphSummary();
    }

    // Free memory
    cudaFree(d_numDigits);
    cudaFree(d_digitsOffsets);
    cudaFree(d_adjList);
    cudaFree(d_adjListSizes);
    cudaFree(d_Seeds);
    cudaFree(d_numSeeds);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_energies);
    cudaFree(d_flatWeights);
    cudaFree(d_neighborsTotClE);
    cudaFree(d_isMergedPi0);
    cudaFree(d_expandedMergedPi0Neighbors);
    cudaFree(d_expandedMergedPi0NumNeighbors);
    cudaFree(d_expandedMergedPi0Weights);
    cudaFree(d_numMergedPi0);
    cudaFree(d_Pi0Indexes);
}

