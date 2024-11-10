
#include "processGraph.h"

__global__ void addNodeToGraphCUDA(int* adjList, int* adjListSizes, int* Nodes, int* numNodes, int maxNodes,
                                   const int* rows, const int* cols, const int* energies, int numDigits, int* flatWeights) {
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

            // Assign the new node in the list of Nodes. TODO: coalesce memory access everywhere
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
            int offset = NodeIncr * 8 * 3;  // Adjusted index for neighbors
            
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
                            
                            flatWeights[NodeIncr * 8 + numNeighbors] = 1; // Assign weight of 1 for now

                            // Assign weight based on distance (for example)
                            // flatWeights[offset / 4] = abs(rows[idx] - neighborRow) + abs(cols[idx] - neighborCol);  // Calculate weight based on Manhattan distance

                            // Increment the offset for the next neighbor
                            offset += 3;

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

// __device__ bool isLocalMaxima(int row, int col, int (*caloValues)[64]) {
//     int neighborOffsets[8][2] = {
//         {-1, -1}, {-1, 0}, {-1, 1}, // Top-left, Top, Top-right
//         {0, -1},          {0, 1},  // Left,       Right
//         {1, -1}, {1, 0}, {1, 1}    // Bottom-left, Bottom, Bottom-right
//     };

//     int neighborRow = row + neighborOffsets[threadIdx.x][0];
//     int neighborCol = col + neighborOffsets[threadIdx.x][1];

//     bool neighborIsGreater = false;

//     if (neighborRow >= 0 && neighborRow < 58 && neighborCol >= 0 && neighborCol < 64) {
//         if (caloValues[neighborRow * 64 + neighborCol] > caloValues[row * 64 + col]) {
//             neighborIsGreater = true;
//         }
//     }

//     // check if any of the neighbors is greater using warp __any_sync, with threads working in groups of 8
//     // Define masks for 8-thread subgroups within a warp
//     unsigned subgroupMask;
//     int lane_id = threadIdx.x % 32;  // Lane within the warp
    
//     if      (lane_id < 8)  subgroupMask = 0x000000FF;
//     else if (lane_id < 16) subgroupMask = 0x0000FF00;
//     else if (lane_id < 24) subgroupMask = 0x00FF0000;
//     else                   subgroupMask = 0xFF000000;
    
//     // Check if any thread in the 8-thread subgroup has predicate == true
//     return __any_sync(subgroupMask, neighborIsGreater);
// }

__global__ void addNodeToGraphCUDAv2(int* adjList, int* adjListSizes, int* Seeds, int* numSeeds, int maxNodes,
                                   const int* rows, const int* cols, const int* energies, int numDigits, int* flatWeights) {

    int localThreadId = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ int caloValues[58][64];

    // create local calo values
    
    // first initialize all values to 0
    for (int i = localThreadId; i < 58*64; i += blockDim.x * blockDim.y) {
        caloValues[i / 64][i % 64] = 0;
    }

    __syncthreads();

    // now fill in the calo values
    for (int i = localThreadId; i < numDigits; i += blockDim.x * blockDim.y) {
        caloValues[rows[i]][cols[i]] = energies[i];
    }

    __syncthreads();

    int digitToProcess = blockIdx.x * blockDim.y + threadIdx.y;

    // Ensure we do not exceed the number of Digits
    if (digitToProcess < numDigits) {
        
        int row = rows[digitToProcess];
        int col = cols[digitToProcess];
        int energy = caloValues[row][col];

        // Check if the energy is greater than threshold = 50. Otherwise, it cant be a seed
        if (!(energy > threshold)) {
            return;
        }

        // Check if it is a local maxima. Otherwise, it cant be a seed
        int neighborOffsets[8][2] = {
            {-1, -1}, {-1, 0}, {-1, 1}, // Top-left, Top, Top-right
            {0, -1},          {0, 1},  // Left,       Right
            {1, -1}, {1, 0}, {1, 1}    // Bottom-left, Bottom, Bottom-right
        };

        int neighborRow = row + neighborOffsets[threadIdx.x][0];
        int neighborCol = col + neighborOffsets[threadIdx.x][1];

        bool neighborIsGreater = false;

        if (neighborRow >= 0 && neighborRow < 58 && neighborCol >= 0 && neighborCol < 64) {
            if (caloValues[neighborRow][neighborCol] > caloValues[row][col]) {
                neighborIsGreater = true;
            }
        }

        // check if any of the neighbors is greater using warp __any_sync, with threads working in groups of 8
        // Define masks for 8-thread subgroups within a warp
        unsigned subgroupMask;
        
        if      (threadIdx.y % 4 == 0) subgroupMask = 0x000000FF;
        else if (threadIdx.y % 4 == 1) subgroupMask = 0x0000FF00;
        else if (threadIdx.y % 4 == 2) subgroupMask = 0x00FF0000;
        else                           subgroupMask = 0xFF000000;
        
        // Check if any thread in the 8-thread subgroup has predicate == true
        if (__any_sync(subgroupMask, neighborIsGreater)) {
            return; // some neighbor is greater, so this is not a local maxima
        }

        int seedNumber = 0;

        if (threadIdx.x == 0) {
            seedNumber = atomicAdd(numSeeds, 1);
        }

        seedNumber = __shfl_sync(subgroupMask, seedNumber, threadIdx.y % 4 * 8);

        // Ensure the maximum number of nodes is not exceeded. This is not needed.
        if (seedNumber >= maxNodes) {
            printf("Error: The maximum number of nodes was exceeded.\n");
            return;
        }

        __shared__ int numNeighbors[32];
        if (threadIdx.x == 0) {
            numNeighbors[threadIdx.y] = 0;
            Seeds[seedNumber * 3] = row;
            Seeds[seedNumber * 3 + 1] = col;
            Seeds[seedNumber * 3 + 2] = energy;
        }

        __syncwarp();

        if (caloValues[neighborRow][neighborCol] > 0) { // neighbor has some energy
            int neighborIdx = atomicAdd(&numNeighbors[threadIdx.y], 1);
            int offset = seedNumber * 8 * 3 + neighborIdx * 3;
            adjList[offset] = neighborRow;
            adjList[offset + 1] = neighborCol;
            adjList[offset + 2] = caloValues[neighborRow][neighborCol];
            flatWeights[seedNumber * 8 + neighborIdx] = 1;
        }

        __syncwarp();

        if (threadIdx.x == 0) {
            adjListSizes[seedNumber] = numNeighbors[threadIdx.y];
        }
    }
}



