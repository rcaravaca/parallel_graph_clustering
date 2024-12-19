// src/main.cu
#include <iostream>
#include "graph.h"
#include "utils.h"
#include "Digit.h"
#include "kernels/insertNodesAndEdges.h"
#include "kernels/processGraph.h"
#include "GraphClustering.h"
#include "GraphClusteringCPU.h"
#include <thread>
#include <chrono>

void n_events_cpu() {
    // Call the function to read the JSON file
    std::vector<Event> events = readJSON("data/digits_values_10.json");
    Event event = events[0];

    std::vector<Digit> digits = event.digits;
    removeDuplicatesAndNegatives(digits);

    std::vector<Digit> digits1000Events;
    std::vector<int> digitsOffsets;
    std::vector<int> numDigits;
    std::vector<Graph> graphs;

    for (int i = 0; i < 1000; i++) {
        digitsOffsets.push_back(digits1000Events.size()); // Capture the current size before appending
        digits1000Events.insert(digits1000Events.end(), digits.begin(), digits.end());
        numDigits.push_back(digits.size());
        graphs.push_back(Graph());
    }

    printf("Number of digits: %lu\n", digits1000Events.size());
    printf("Number of digitsOffsets: %lu\n", digitsOffsets.size());
    printf("Number of numDigits: %lu\n", numDigits.size());
    
    // check running time
    auto startcpu = std::chrono::high_resolution_clock::now();
    GraphInsertionNEventsCPU(graphs, 6016, digits1000Events, digitsOffsets, numDigits);
    auto endcpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endcpu - startcpu;
    std::cout << "Elapsed time CPU: " << elapsed.count() << " seconds" << std::endl;

    // for (int i = 0; i < 1000; i++) {
    //     graphs[i].GraphSummary();
    // }
}

void n_events_gpu() {
    // Call the function to read the JSON file
    std::vector<Event> events = readJSON("data/digits_values_10.json");
    Event event = events[0];

    std::vector<Digit> digits = event.digits;
    removeDuplicatesAndNegatives(digits);

    std::vector<Digit> digits1000Events;
    std::vector<int> digitsOffsets;
    std::vector<int> numDigits;
    std::vector<Graph> graphs;

    for (int i = 0; i < 1000; i++) {
        digitsOffsets.push_back(digits1000Events.size()); // Capture the current size before appending
        digits1000Events.insert(digits1000Events.end(), digits.begin(), digits.end());
        numDigits.push_back(digits.size());
        graphs.push_back(Graph());
    }

    printf("Number of digits: %lu\n", digits1000Events.size());
    printf("Number of digitsOffsets: %lu\n", digitsOffsets.size());
    printf("Number of numDigits: %lu\n", numDigits.size());

    // check running time
    auto startcpu = std::chrono::high_resolution_clock::now();
    GraphInsertionNEventsV1(graphs, 6016, digits1000Events, digitsOffsets, numDigits);
    auto endcpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endcpu - startcpu;
    std::cout << "Elapsed time GPU: " << elapsed.count() << " seconds" << std::endl;

    for (int i = 0; i < 1000; i++) {
        graphs[i].GraphSummary();
    }
}

void n_events_gpu_with_pi0s() {
    // Call the function to read the JSON file
    std::vector<Event> events = readJSON("data/digits_values_10.json");
    Event event = events[0];

    std::vector<Digit> digits = event.digits;
    removeDuplicatesAndNegatives(digits);

    std::vector<Digit> digits1000Events;
    std::vector<int> digitsOffsets;
    std::vector<int> numDigits;
    std::vector<Graph> graphs;

    for (int i = 0; i < 1000; i++) {
        digitsOffsets.push_back(digits1000Events.size()); // Capture the current size before appending
        digits1000Events.insert(digits1000Events.end(), digits.begin(), digits.end());
        numDigits.push_back(digits.size());
        graphs.push_back(Graph());
    }

    printf("Number of digits: %lu\n", digits1000Events.size());
    printf("Number of digitsOffsets: %lu\n", digitsOffsets.size());
    printf("Number of numDigits: %lu\n", numDigits.size());

    // check running time
    auto startcpu = std::chrono::high_resolution_clock::now();
    GraphInsertionNEventsWithPi0V1(graphs, 6016, digits1000Events, digitsOffsets, numDigits);
    auto endcpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endcpu - startcpu;
    std::cout << "Elapsed time GPU: " << elapsed.count() << " seconds" << std::endl;

    for (int i = 0; i < 1000; i++) {
        // graphs[i].GraphSummary();
    }
}

void n_events_gpu_base_version() {
    // Call the function to read the JSON file
    std::vector<Event> events = readJSON("data/digits_values_10.json");
    Event event = events[0];

    std::vector<Digit> digits = event.digits;
    removeDuplicatesAndNegatives(digits);

    std::vector<Digit> digits1000Events;
    std::vector<int> digitsOffsets;
    std::vector<int> numDigits;
    std::vector<Graph> graphs;

    for (int i = 0; i < 1000; i++) {
        digitsOffsets.push_back(digits1000Events.size()); // Capture the current size before appending
        digits1000Events.insert(digits1000Events.end(), digits.begin(), digits.end());
        numDigits.push_back(digits.size());
        graphs.push_back(Graph());
    }

    printf("Number of digits: %lu\n", digits1000Events.size());
    printf("Number of digitsOffsets: %lu\n", digitsOffsets.size());
    printf("Number of numDigits: %lu\n", numDigits.size());

    // check running time
    auto startcpu = std::chrono::high_resolution_clock::now();
    GraphInsertionNEventsBase(graphs, 6016, digits1000Events, digitsOffsets, numDigits);
    auto endcpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endcpu - startcpu;
    std::cout << "Elapsed time GPU base version: " << elapsed.count() << " seconds" << std::endl;

    for (int i = 0; i < 1000; i++) {
        graphs[i].GraphSummary();
    }
}

int main() {

    // n_events_gpu();
    n_events_gpu_with_pi0s();
    // n_events_gpu_base_version();
    // n_events_cpu();

    return 0;

    // Call the function to read the JSON file
    std::vector<Event> events = readJSON("data/digits_values_10.json");
    Event event = events[0];

    std::vector<Digit> digits = event.digits;
    removeDuplicatesAndNegatives(digits);

    Graph graph;

    // Print digits summary
    DigitEnergySummary(digits);
 
    // Do the graph nodes insertion
    // GraphInsertion(graph, 6016, digits);
    
    GraphInsertionV2(graph, 6016, digits);

    // Print Graph Nodes summary
    graph.GraphSummary();

    graph.printGraph();

    // Check for duplicate IDs
    bool hasDuplicates = graph.checkForDuplicateIDs();

    if (hasDuplicates) {
        std::cout << "There are duplicate IDs in the graph." << std::endl;
    } else {
        std::cout << "No duplicate IDs found in the graph." << std::endl;
    }

    return 0;
}