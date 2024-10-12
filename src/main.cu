// src/main.cu
#include <iostream>
#include "graph.h"
#include "utils.h"
#include "Digit.h"
#include "kernels/insertNodesAndEdges.h"
#include "kernels/processGraph.h"
#include "GraphClustering.h"

int main() {

    // Call the function to read the JSON file
    std::vector<Event> events = readJSON("data/digits_values_10.json");
    Event event = events[1];

    std::vector<Digit> digits = event.digits;

    // Create a graph with 4 nodes
    Graph graph;


    // Digit A(0, 0, 5);  // Nodo A (fila 0, columna 0, energía 5)
    // Digit B(0, 1, 3);  // Nodo B (fila 0, columna 1, energía 3)
    // Digit C(1, 0, 8);  // Nodo C (fila 1, columna 0, energía 8)
    // Digit D(1, 1, 2);  // Nodo D (fila 1, columna 1, energía 2)

    // graph.addNode(A);
    // graph.addNode(B);
    // graph.addNode(C);
    // graph.addNode(D);

    // graph.addEdge(A, B);  // A -> B
    // graph.addEdge(A, C);  // A -> C
    // graph.addEdge(B, D);  // B -> D
    // graph.addEdge(C, D);  // C -> D

    // std::cout << "Adjacency List of the Graph:" << std::endl;
    // graph.printGraph();

    // Print digits summary
    DigitEnergySummary(digits);
 
    // Do the graph nodes insertion
    GraphInsertion(graph, 6016, digits);

    // Print Graph Nodes summary
    graph.GraphSummary();


    return 0;
}

// int main(int argc, char* argv[]) {

    // // Call the function to read the JSON file
    // std::vector<Event> events = readJSON("data/digits_values_10.json");

    // Iterate over the events and display their digits
    // for (const auto& event : events) {
    //     std::cout << "Event: " << event.event_id << " has " << event.digits.size() << " digits count..." << std::endl;
    //     for (const auto& digit : event.digits) {
    //         std::cout << "\tDigit - ID: " << digit.getID() << "\tDigit - Row: " << digit.getRow() << ", Col: " << digit.getCol() << ", Energy: " << digit.getEnergy() << std::endl;
    //     }
    // }

    // Event event = events[0];

    // // Initialize the graph
    // Graph graph(4);

//     // Add edges (A -> B, A -> C, B -> D, C -> D)
//     graph.addEdge(0, 1);  // A -> B
//     graph.addEdge(0, 2);  // A -> C
//     graph.addEdge(1, 3);  // B -> D
//     graph.addEdge(2, 3);  // C -> D

//     // Print the graph (for debugging)
//     graph.printGraph();


    // // Load the graph data
    // int numNodes = event.digits.size();  // node count
    // initializeGraph(graph, numNodes);

    // // In practice, you might load this from the input_data.txt in the data directory
    // int* energies = new int[numNodes];
    // int* ids = new int[numNodes];
    // initializeEnergiesAndIds(energies, ids, numNodes);

    // // Allocate memory for the GPU (assuming the adjacency list and offset arrays are already filled)
    // cudaMalloc(&(graph.adjList), graph.numNodes * graph.numNodes * sizeof(int));  // Size
    // cudaMalloc(&(graph.offset), graph.numNodes * sizeof(int));

    // // Initialize the adjacency list and offsets
    // initializeAdjListAndOffsets(graph);

    // // Print the graph
    // printGraph(graph, 10);

    // // Allocate additional arrays on the GPU
    // int *d_energies, *d_ids, *d_mergedPi0;
    // cudaMalloc(&d_energies, numNodes * sizeof(int));
    // cudaMalloc(&d_ids, numNodes * sizeof(int));
    // cudaMalloc(&d_mergedPi0, numNodes * sizeof(int));

    // // Copy data to GPU
    // cudaMemcpy(d_energies, energies, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_ids, ids, numNodes * sizeof(int), cudaMemcpyHostToDevice);

    // // Define grid and block size
    // int blockSize = 256;
    // int numBlocks = (numNodes + blockSize - 1) / blockSize;

    // // Launch the kernel
    // //insertNodesAndEdges<<<numBlocks, blockSize>>>(d_energies, d_ids, graph.adjList, graph.offset, d_mergedPi0, numNodes);

    // // Synchronize to ensure the kernel execution is complete
    // cudaDeviceSynchronize();

    // // Free GPU memory
    // cudaFree(d_energies);
    // cudaFree(d_ids);
    // cudaFree(d_mergedPi0);
    // cudaFree(graph.adjList);
    // cudaFree(graph.offset);

    // // Free host memory
    // delete[] energies;
    // delete[] ids;

//     std::cout << "Parallel Graph Clustering executed successfully." << std::endl;
//     return 0;
// }
