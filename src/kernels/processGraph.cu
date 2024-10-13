
#include "processGraph.h"

__global__ void addNodeToGraphCUDA(int* adjList, int* adjListSizes, int* Nodes, int* numNodes, int maxNodes,
                                   const int* rows, const int* cols, const int* energies, int numDigits,
                                   int* flatWeights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not exceed the number of Digits
    if (idx < numDigits) {

        // Check if the energy is greater than threshold = 50
        if (energies[idx] > threshold) {
            
            // Get a safe node increment using atomicAdd
            int NodeIncr = atomicAdd(numNodes, 1);
            
            // Ensure the maximum number of nodes is not exceeded
            if (NodeIncr >= maxNodes) {
                printf("Error: The maximum number of nodes was exceeded.\n");
                return;
            }

            // Assign the new node in the list of Nodes
            Nodes[NodeIncr * 3] = rows[idx];       
            Nodes[NodeIncr * 3 + 1] = cols[idx];   
            Nodes[NodeIncr * 3 + 2] = energies[idx];  

            // Initialize the adjacency list for this node (no neighbors yet)
            adjListSizes[NodeIncr] = 0;

            // Add valid neighbors (up to 8 possible) based on row, col offsets
            int neighbors[8][2] = {
                {0, -1},  // Left (same row, col - 1)
                {0, 1},   // Right (same row, col + 1)
                {-1, 0},  // Up (row - 1, same col)
                {1, 0},   // Down (row + 1, same col)
                {-1, -1}, // Upper left diagonal (row - 1, col - 1)
                {-1, 1},  // Upper right diagonal (row - 1, col + 1)
                {1, -1},  // Lower left diagonal (row + 1, col - 1)
                {1, 1}    // Lower right diagonal (row + 1, col + 1)
            };

            // Iterate over the potential neighbors
            // for (int neighborIdx = 0; neighborIdx < 8; ++neighborIdx) {
            //     int neighborRow = rows[idx] + neighbors[neighborIdx][0];  // Calculate neighbor row
            //     int neighborCol = cols[idx] + neighbors[neighborIdx][1];  // Calculate neighbor col

            //     // Find the neighbor in the input arrays (assuming sorted or accessible by index)
            //     for (int j = 0; j < numDigits; ++j) {
            //         if (rows[j] == neighborRow && cols[j] == neighborCol) {
            //             int adjIndex = atomicAdd(&adjListSizes[NodeIncr], 1);  // Increment adjacency list size

            //             if (energies[j] > 0) {
            //             // Store the neighbor in the adjacency list
            //                 adjList[NodeIncr * 8 + adjIndex * 4] = neighborRow;
            //                 adjList[NodeIncr * 8 + adjIndex * 4 + 1] = neighborCol;
            //                 adjList[NodeIncr * 8 + adjIndex * 4 + 2] = energies[j];  // Store neighbor energy

            //                 // Assign weight (for example, based on distance or constant)
            //                 flatWeights[NodeIncr * 8 + adjIndex] = 1;  // Example constant weight
            //             }
            //         }
            //     }
            // }

        }
    }
}



