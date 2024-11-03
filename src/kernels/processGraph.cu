
#include "processGraph.h"

__global__ void addNodeToGraphCUDA(int* adjList, int* adjListSizes, int* Nodes, int* numNodes, int maxNodes,
                                   const int* rows, const int* cols, const int* energies, int numDigits) {
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

            // Add valid neighbors (up to 8 possible) based on row, col offsets. TODO: move this to constant memory
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

            int numNeighbors = 0;
            int offset = NodeIncr * 8 * 4;  // Adjusted index for neighbors
            
            // Iterate over the potential neighbors
            for (int neighborIdx = 0; neighborIdx < 8; ++neighborIdx) {
                int neighborRow = rows[idx] + neighbors[neighborIdx][0];  // Calculate neighbor row
                int neighborCol = cols[idx] + neighbors[neighborIdx][1];  // Calculate neighbor col
                if (neighborRow < 0 || neighborCol < 0 ) { // TODO: which is the upper bound?
                    continue;  // Skip invalid neighbors that would be out of bounds
                }

                // Find the neighbor in the input arrays (assuming sorted or accessible by index)
                for (int j = 0; j < numDigits; ++j) {
                    if (rows[j] == neighborRow && cols[j] == neighborCol) {
                        
                        if (energies[j] > 0) {

                            // Store the neighbor in the adjacency list
                            adjList[offset] = neighborRow;
                            adjList[offset + 1] = neighborCol;
                            adjList[offset + 2] = energies[j];  // Store neighbor energy
                            adjList[offset + 3] = 1; // Assign weight of 1 for now

                            // Assign weight based on distance (for example)
                            // flatWeights[offset / 4] = abs(rows[idx] - neighborRow) + abs(cols[idx] - neighborCol);  // Calculate weight based on Manhattan distance

                            // Increment the offset for the next neighbor
                            offset += 4;

                            // Increment the number of neighbors for this node
                            numNeighbors++;

                        }
                        // Break out of the inner loop once the neighbor is found. Doesnt matter if it was added or not
                        break;
                    }
                }
            }

            // Update the number of neighbors for this node
            adjListSizes[NodeIncr] = numNeighbors;
        }
    }
}



