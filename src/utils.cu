// src/utils.cu
#include "utils.h"
#include <cstdlib>
#include <ctime>

/**
 * @brief Initializes the energies and ids arrays with example data.
 *
 * @param energies Pointer to the array that will store energy values for each node.
 * @param ids Pointer to the array that will store IDs corresponding to each node.
 * @param numNodes The total number of nodes in the graph.
 */
void initializeEnergiesAndIds(int* energies, int* ids, int numNodes) {
    // Seed the random number generator for reproducibility
    std::srand(std::time(0));

    // Initialize energies and ids with some example data
    for (int i = 0; i < numNodes; i++) {
        energies[i] = std::rand() % 100; // Random energy value between 0 and 99
        ids[i] = i;  // Assign the node's index as its ID
    }
}
