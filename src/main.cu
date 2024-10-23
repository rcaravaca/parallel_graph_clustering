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
    Event event = events[0];

    std::vector<Digit> digits = event.digits;
    removeDuplicatesAndNegatives(digits);

    Graph graph;

    // Print digits summary
    DigitEnergySummary(digits);
 
    // Do the graph nodes insertion
    GraphInsertion(graph, 6016, digits);

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