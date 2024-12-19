#include "GraphClusteringCPU.h"
#include <algorithm>

void graphInsertion(Graph& graph, int maxNodes, std::vector<Digit>& digits) {
    
    std::sort(digits.begin(), digits.end(), [](const Digit& a, const Digit& b) {
        return a.getEnergy() > b.getEnergy();
    });

    // std::vector<int> mergedPi0;
    // mergedPi0.reserve( 1024 );

    std::map<int, int> outEdges;


    std::vector<std::vector<int>> matrix(58, std::vector<int>(64, 0));
    for (const Digit& digit : digits) {
        matrix[digit.getRow()][digit.getCol()] = digit.getEnergy();
    }

    int neighborOffsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, // Top-left, Top, Top-right
        {0, -1},          {0, 1},  // Left,       Right
        {1, -1}, {1, 0}, {1, 1}    // Bottom-left, Bottom, Bottom-right
    };
    // Insert nodes into the graph
    for (const Digit& digit : digits) {
        if (digit.getEnergy() <= 50) {
            continue;
        }
        bool isLocalMax = true;
        for (int i = 0; i < 8; i++) {
            int row = digit.getRow() + neighborOffsets[i][0];
            int col = digit.getCol() + neighborOffsets[i][1];
            if (row >= 0 && row < 58 && col >= 0 && col < 64) {
                if (matrix[row][col] > digit.getEnergy()) {
                    isLocalMax = false;
                    break;
                }
            }
        }

        if (isLocalMax) {
            // Add the node to the graph
            graph.addNode(digit);
            
            for (int i = 0; i < 8; i++) {
                int row = digit.getRow() + neighborOffsets[i][0];
                int col = digit.getCol() + neighborOffsets[i][1];
                if (row >= 0 && row < 58 && col >= 0 && col < 64) {
                    if (matrix[row][col] > 0) {
                        graph.addEdge(digit, Digit(row, col, matrix[row][col]), 1);
                    }
                }
            }
        }
    }
}

void GraphInsertionNEventsCPU(
    std::vector<Graph>& graphs,
    const int maxSeeds,
    const std::vector<Digit>& digits,
    const std::vector<int>& digitsOffsets,
    const std::vector<int>& numDigits) {

    // Assume number of events == number of graphs == size of digitsOffsets == size of numDigits

    if (graphs.size() != digitsOffsets.size() || graphs.size() != numDigits.size()) {
        std::cerr << "GraphInsertionNEventsCPU: Error: Number of graphs, digitsOffsets, and numDigits do not match." << std::endl;
        return;
    }

    int numEvents = graphs.size();

    for (int i = 0; i < numEvents; i++) {
        int start = digitsOffsets[i];
        int end = start + numDigits[i];

        std::vector<Digit> eventDigits(digits.begin() + start, digits.begin() + end);

        graphInsertion(graphs[i], maxSeeds, eventDigits);
    }

}