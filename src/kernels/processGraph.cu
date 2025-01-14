
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

__global__ void addNodeToGraphCUDANEventsBase(int* numDigits, int* digitsOffsets, int* adjList, int* adjListSizes, int* Seeds, int* numSeeds, int maxSeeds, const int* rows, const int* cols, const int* energies, int* flatWeights) {
    int eventIdx = blockIdx.x;

    int digitsInEvent = numDigits[eventIdx];

    rows = rows + digitsOffsets[eventIdx];
    cols = cols + digitsOffsets[eventIdx];
    energies = energies + digitsOffsets[eventIdx];


    adjList = adjList + eventIdx * maxSeeds * 8 * 3;
    adjListSizes = adjListSizes + eventIdx * maxSeeds;
    Seeds = Seeds + eventIdx * maxSeeds * 3;
    flatWeights = flatWeights + eventIdx * maxSeeds * 8;

    // Ensure we do not exceed the number of Digits
    for (int idx = threadIdx.x; idx < digitsInEvent; idx += blockDim.x) {

        // Check if the energy is greater than threshold = 50
        if (energies[idx] > threshold) {
            
            // Get a safe node increment using atomicAdd
            int NodeIncr = atomicAdd(&numSeeds[eventIdx], 1);
            
            // Ensure the maximum number of nodes is not exceeded
            if (NodeIncr >= maxSeeds) {
                printf("Error: The maximum number of nodes was exceeded.\n");
                return;
            }

            // Assign the new node in the list of Nodes. TODO: coalesce memory access everywhere
            Seeds[NodeIncr * 3] = rows[idx];
            Seeds[NodeIncr * 3 + 1] = cols[idx];   
            Seeds[NodeIncr * 3 + 2] = energies[idx];

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
                for (int j = 0; j < digitsInEvent; ++j) {
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

// 8x32 threads per block
// process 32 digits at a time, using 8 threads per digit
__global__ void addNodeToGraphCUDANEventsV1(int* numDigits, int* digitsOffsets, int* adjList, int* adjListSizes, int* Seeds, int* numSeeds, int maxSeeds, const int* rows, const int* cols, const int* energies, int* flatWeights) {

    int eventIdx = blockIdx.x;

    __shared__ int caloValues[58][64];

    int localThreadId = threadIdx.y * blockDim.x + threadIdx.x;

        // first initialize all values to 0
    for (int i = localThreadId; i < 58*64; i += blockDim.x * blockDim.y) {
        caloValues[i / 64][i % 64] = 0;
    }

    __syncthreads();

    int digitsInEvent = numDigits[eventIdx];

    rows = rows + digitsOffsets[eventIdx];
    cols = cols + digitsOffsets[eventIdx];
    energies = energies + digitsOffsets[eventIdx];

    // now fill in the calo values
    for (int i = localThreadId; i < digitsInEvent; i += blockDim.x * blockDim.y) {
        caloValues[rows[i]][cols[i]] = energies[i];
    }

    __syncthreads();

    adjList = adjList + eventIdx * maxSeeds * 8 * 3;
    adjListSizes = adjListSizes + eventIdx * maxSeeds;
    Seeds = Seeds + eventIdx * maxSeeds * 3;
    flatWeights = flatWeights + eventIdx * maxSeeds * 8;

    for (int digit = threadIdx.y; digit < digitsInEvent; digit += blockDim.y) {

        int row = rows[digit];
        int col = cols[digit];
        int energy = caloValues[row][col];

        // Check if the energy is greater than threshold = 50. Otherwise, it cant be a seed
        if (!(energy > threshold)) {
            continue;
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
        
        // Check if any thread in the 8-thread subgroup has predicate == trueº
        if (__any_sync(subgroupMask, neighborIsGreater)) {
            continue; // some neighbor is greater, so this is not a local maxima
        }

        int seedNumber = 0;

        if (threadIdx.x == 0) {
            seedNumber = atomicAdd(&numSeeds[eventIdx], 1); // TODO: this could be done now in shared memory as each event is processed in a single block
        }

        seedNumber = __shfl_sync(subgroupMask, seedNumber, threadIdx.y % 4 * 8);

        // Ensure the maximum number of nodes is not exceeded. This is not needed.
        if (seedNumber >= maxSeeds) {
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

// 8x32 threads per block
// process 32 digits at a time, using 8 threads per digit
__global__ void addNodeToGraphCUDANEventsWithMergedPi0V1(int* numDigits, int* digitsOffsets, int* adjList, int* adjListSizes, int* Seeds, int* numSeeds, int maxSeeds, const int* rows, const int* cols, const int* energies, int* flatWeights, int8_t* isMergedPi0, int* numMergedPi0s) {

    int eventIdx = blockIdx.x;

    __shared__ int caloValues[58][64];

    int localThreadId = threadIdx.y * blockDim.x + threadIdx.x;

        // first initialize all values to 0
    for (int i = localThreadId; i < 58*64; i += blockDim.x * blockDim.y) {
        caloValues[i / 64][i % 64] = 0;
    }

    __syncthreads();

    int digitsInEvent = numDigits[eventIdx];

    rows = rows + digitsOffsets[eventIdx];
    cols = cols + digitsOffsets[eventIdx];
    energies = energies + digitsOffsets[eventIdx];

    // now fill in the calo values
    for (int i = localThreadId; i < digitsInEvent; i += blockDim.x * blockDim.y) {
        caloValues[rows[i]][cols[i]] = energies[i];
    }

    __syncthreads();

    adjList = adjList + eventIdx * maxSeeds * 8 * 3;
    adjListSizes = adjListSizes + eventIdx * maxSeeds;
    Seeds = Seeds + eventIdx * maxSeeds * 3;
    flatWeights = flatWeights + eventIdx * maxSeeds * 8;

    isMergedPi0 = isMergedPi0 + eventIdx * maxSeeds;

    for (int digit = threadIdx.y; digit < digitsInEvent; digit += blockDim.y) {

        int row = rows[digit];
        int col = cols[digit];
        int energy = caloValues[row][col];

        // Check if the energy is greater than threshold = 50. Otherwise, it cant be a seed
        if (!(energy > threshold)) {
            continue;
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
        int neighborEnergy = 0;

        if (neighborRow >= 0 && neighborRow < 58 && neighborCol >= 0 && neighborCol < 64) {
            neighborEnergy = caloValues[neighborRow][neighborCol];
            if (neighborEnergy > caloValues[row][col]) {
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
            continue; // some neighbor is greater, so this is not a local maxima
        }

        int seedNumber = 0;

        if (threadIdx.x == 0) {
            seedNumber = atomicAdd(&numSeeds[eventIdx], 1); // TODO: this could be done now in shared memory as each event is processed in a single block
        }

        seedNumber = __shfl_sync(subgroupMask, seedNumber, threadIdx.y % 4 * 8);

        // Ensure the maximum number of nodes is not exceeded. This is not needed.
        if (seedNumber >= maxSeeds) {
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

        __syncwarp();

        int maxNeighborEnergy = neighborEnergy;
        for (int offset = 4; offset > 0; offset /= 2) { // Reduce within 8 threads
            int otherVal = __shfl_down_sync(subgroupMask, maxNeighborEnergy, offset);
            maxNeighborEnergy = fmaxf(maxNeighborEnergy, otherVal);
        }

        __syncwarp();
        maxNeighborEnergy = __shfl_sync(subgroupMask, maxNeighborEnergy, threadIdx.y % 4 * 8); // Broadcast to all threads

        //Identify threads with the maximum value
        unsigned int maxMask = __ballot_sync(subgroupMask, neighborEnergy == maxNeighborEnergy);

        // Get the first thread index within the subgroup
        int maxThreadInGroup = (__ffs(maxMask & subgroupMask) - 1) % 8; // Local index within warp

        // Check if neighbor could be a merged Pi0. Main seed energy > 1000 and neighbor energy > 25% main seed energy
        if (threadIdx.x == maxThreadInGroup) {
            if (caloValues[row][col] > 1000 && caloValues[neighborRow][neighborCol] > 0.25 * caloValues[row][col]) {
                int mergedPi0Idx = atomicAdd(&numMergedPi0s[eventIdx], 1); // TODO: value not used
                isMergedPi0[seedNumber] = threadIdx.x; // position will be known with constant values defined in the header
            }
        }
    }
}

// 8x32 threads per block
// process 32 digits at a time, using 8 threads per digit
__global__ void addNodeToGraphCUDANEventsWithMergedPi0V2(int* numDigits, int* digitsOffsets, int* adjList, int* adjListSizes, int* Seeds, int* numSeeds, int maxSeeds, const int* rows, const int* cols, const int* energies, int* neighborsTotClE, int8_t* isMergedPi0, int* numMergedPi0s, int* overlapTracking) {

    int eventIdx = blockIdx.x;

    __shared__ int caloValues[58][64];

    int localThreadId = threadIdx.y * blockDim.x + threadIdx.x;

        // first initialize all values to 0
    for (int i = localThreadId; i < 58*64; i += blockDim.x * blockDim.y) {
        caloValues[i / 64][i % 64] = 0;
    }

    __syncthreads();

    int digitsInEvent = numDigits[eventIdx];

    rows = rows + digitsOffsets[eventIdx];
    cols = cols + digitsOffsets[eventIdx];
    energies = energies + digitsOffsets[eventIdx];

    // now fill in the calo values
    for (int i = localThreadId; i < digitsInEvent; i += blockDim.x * blockDim.y) {
        caloValues[rows[i]][cols[i]] = energies[i];
    }

    __syncthreads();

    adjList = adjList + eventIdx * maxSeeds * 8 * 3;
    adjListSizes = adjListSizes + eventIdx * maxSeeds;
    Seeds = Seeds + eventIdx * maxSeeds * 3;
    neighborsTotClE = neighborsTotClE + eventIdx * 58 * 64; // should this be 58 * 64? I guess but eventually this will be bigger to reach 6016
    overlapTracking = overlapTracking + eventIdx * 58 * 64;

    isMergedPi0 = isMergedPi0 + eventIdx * maxSeeds;

    for (int digit = threadIdx.y; digit < digitsInEvent; digit += blockDim.y) {

        int row = rows[digit];
        int col = cols[digit];
        int energy = caloValues[row][col];

        // Check if the energy is greater than threshold = 50. Otherwise, it cant be a seed
        if (!(energy > threshold)) {
            continue;
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
        int neighborEnergy = 0;

        if (neighborRow >= 0 && neighborRow < 58 && neighborCol >= 0 && neighborCol < 64) {
            neighborEnergy = caloValues[neighborRow][neighborCol];
            if (neighborEnergy > caloValues[row][col]) {
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
            continue; // some neighbor is greater, so this is not a local maxima
        }

        // clusters with 0 neighbors are not interesting
        if (__popc(__ballot_sync(subgroupMask, neighborEnergy > 0)) < 1) {
            continue;
        }

        int seedNumber = 0;

        if (threadIdx.x == 0) {
            seedNumber = atomicAdd(&numSeeds[eventIdx], 1); // TODO: this could be done now in shared memory as each event is processed in a single block
        }

        seedNumber = __shfl_sync(subgroupMask, seedNumber, threadIdx.y % 4 * 8);

        // Ensure the maximum number of nodes is not exceeded. This is not needed.
        if (seedNumber >= maxSeeds) {
            printf("Error: The maximum number of nodes was exceeded.\n");
            return;
        }

        // get total cluster energy
        int totalClusterEnergy = neighborEnergy;
        for (int offset = 4; offset > 0; offset /= 2) {
            totalClusterEnergy += __shfl_down_sync(subgroupMask, totalClusterEnergy, offset);
        }

        if (threadIdx.x == 0) {
            totalClusterEnergy += energy; // add the seed energy
        }

        totalClusterEnergy = __shfl_sync(subgroupMask, totalClusterEnergy, threadIdx.y % 4 * 8); // Broadcast to all threads


        __shared__ int numNeighbors[32];
        if (threadIdx.x == 0) {
            numNeighbors[threadIdx.y] = 0;
            Seeds[seedNumber * 3] = row;
            Seeds[seedNumber * 3 + 1] = col;
            Seeds[seedNumber * 3 + 2] = energy;
        }

        __syncwarp();

        if (neighborRow >= 0 && neighborRow < 58 && neighborCol >= 0 && neighborCol < 64 && 
            caloValues[neighborRow][neighborCol] > 0) { // neighbor has some energy

            int neighborIdx = atomicAdd(&numNeighbors[threadIdx.y], 1);
            int offset = seedNumber * 8 * 3 + neighborIdx * 3;
            adjList[offset] = neighborRow;
            adjList[offset + 1] = neighborCol;
            adjList[offset + 2] = caloValues[neighborRow][neighborCol];
            atomicAdd(&neighborsTotClE[neighborRow * 64 + neighborCol], totalClusterEnergy);
            atomicAdd(&overlapTracking[neighborRow * 64 + neighborCol], 1);
        }

        __syncwarp();

        if (threadIdx.x == 0) {
            adjListSizes[seedNumber] = numNeighbors[threadIdx.y];
            // for later use
            neighborsTotClE[row * 64 + col] = totalClusterEnergy; // seed position will contain the 3x3 cluster energy
        }

        __syncwarp();

        int maxNeighborEnergy = neighborEnergy;
        for (int offset = 4; offset > 0; offset /= 2) { // Reduce within 8 threads
            int otherVal = __shfl_down_sync(subgroupMask, maxNeighborEnergy, offset);
            maxNeighborEnergy = fmaxf(maxNeighborEnergy, otherVal);
        }

        __syncwarp();
        maxNeighborEnergy = __shfl_sync(subgroupMask, maxNeighborEnergy, threadIdx.y % 4 * 8); // Broadcast to all threads

        //Identify threads with the maximum value
        unsigned int maxMask = __ballot_sync(subgroupMask, neighborEnergy == maxNeighborEnergy);

        // Get the first thread index within the subgroup
        int maxThreadInGroup = (__ffs(maxMask & subgroupMask) - 1) % 8; // Local index within warp

        // Check if neighbor could be a merged Pi0. Main seed energy > 1000 and neighbor energy > 25% main seed energy
        if (threadIdx.x == maxThreadInGroup) {
            if (caloValues[row][col] > 1000 && caloValues[neighborRow][neighborCol] > 0.25 * caloValues[row][col]) {
                int mergedPi0Idx = atomicAdd(&numMergedPi0s[eventIdx], 1); // TODO: value not used
                isMergedPi0[seedNumber] = threadIdx.x; // position will be known with constant values defined in the header
            } else {
                isMergedPi0[seedNumber] = -1; // TODO: this default value should be set on host probably
            }
        }
    }
}

__global__ void expandPi0sNeighborsV1(int* numDigits, int* digitsOffsets, const int* rows, const int* cols, const int* energies, int* Seeds, int maxSeeds, int* neighborsTotClE, int* numMergedPi0s, int* mergedPi0Indexes, int8_t* mergedPi0sDirection, int* expandedMergedPi0Neighbors, int* expandedMergedPi0NeighborsSizes, int* overlapTracking) {

    int eventIdx = blockIdx.x;

    __shared__ int caloValues[58][64];

    int localThreadId = threadIdx.y * blockDim.x + threadIdx.x;

        // first initialize all values to 0
    for (int i = localThreadId; i < 58*64; i += blockDim.x * blockDim.y) {
        caloValues[i / 64][i % 64] = 0;
    }

    __syncthreads();

    int digitsInEvent = numDigits[eventIdx];

    rows = rows + digitsOffsets[eventIdx];
    cols = cols + digitsOffsets[eventIdx];
    energies = energies + digitsOffsets[eventIdx];
    neighborsTotClE = neighborsTotClE + eventIdx * 58 * 64;
    overlapTracking = overlapTracking + eventIdx * 58 * 64; 
    Seeds = Seeds + eventIdx * maxSeeds * 3;

    // now fill in the calo values
    for (int i = localThreadId; i < digitsInEvent; i += blockDim.x * blockDim.y) {
        caloValues[rows[i]][cols[i]] = energies[i];
    }

    __syncthreads();

    int mergedPi0s = numMergedPi0s[eventIdx];
    mergedPi0Indexes = mergedPi0Indexes + eventIdx * maxSeeds;
    mergedPi0sDirection = mergedPi0sDirection + eventIdx * maxSeeds;
    
    expandedMergedPi0Neighbors = expandedMergedPi0Neighbors + eventIdx * maxSeeds * 5 * 3;
    expandedMergedPi0NeighborsSizes = expandedMergedPi0NeighborsSizes + eventIdx * maxSeeds;

    uint8_t neighborsToAddByDirection[8] = {
        0b00101111, // TOP_LEFT
        0b00000111, // TOP
        0b10010111, // TOP_RIGHT
        0b00101001, // LEFT
        0b10010100, // RIGHT
        0b11101001, // BOTTOM_LEFT
        0b11100000, // BOTTOM
        0b11110100  // BOTTOM_RIGHT
    };

    int neighborOffsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, // Top-left, Top, Top-right
        {0, -1},           {0, 1},  // Left,       Right
        {1, -1},  {1, 0},  {1, 1}    // Bottom-left, Bottom, Bottom-right
    };

    unsigned subgroupMask;
        
    if      (threadIdx.y % 4 == 0) subgroupMask = 0x000000FF;
    else if (threadIdx.y % 4 == 1) subgroupMask = 0x0000FF00;
    else if (threadIdx.y % 4 == 2) subgroupMask = 0x00FF0000;
    else                           subgroupMask = 0xFF000000;

    for (int pi0 = threadIdx.y; pi0 < mergedPi0s; pi0 += blockDim.y) {
        
        int seedRow = Seeds[mergedPi0Indexes[pi0] * 3];
        int seedCol = Seeds[mergedPi0Indexes[pi0] * 3 + 1];
        // int seedEnergy = Seeds[mergedPi0Indexes[pi0] * 3 + 2];
        int originalClusterEnergy = neighborsTotClE[seedRow * 64 + seedCol];
        int8_t direction = mergedPi0sDirection[mergedPi0Indexes[pi0]];

        int pi0Row = seedRow + neighborOffsets[direction][0];
        int pi0Col = seedCol + neighborOffsets[direction][1];
        int pi0Energy = caloValues[pi0Row][pi0Col];
        
        int neighborEnergy = 0;
        bool addedAsPi0Neighbor = false;
        int neighborRow, neighborCol;

        if (neighborsToAddByDirection[direction] & (1 << threadIdx.x)) {
            neighborRow = pi0Row + neighborOffsets[threadIdx.x][0];
            neighborCol = pi0Col + neighborOffsets[threadIdx.x][1];
            if (neighborRow >= 0 && neighborRow < 58 && neighborCol >= 0 && neighborCol < 64) {
                neighborEnergy = caloValues[neighborRow][neighborCol];
                if (neighborEnergy > 0 && neighborEnergy < pi0Energy) {
                    addedAsPi0Neighbor = true;
                    
                    int neighborIdx = atomicAdd(&expandedMergedPi0NeighborsSizes[mergedPi0Indexes[pi0]], 1);
                    int offset = mergedPi0Indexes[pi0] * 5 * 3 + neighborIdx * 3;
                    expandedMergedPi0Neighbors[offset] = neighborRow;
                    expandedMergedPi0Neighbors[offset + 1] = neighborCol;
                    expandedMergedPi0Neighbors[offset + 2] = neighborEnergy;
                    atomicAdd(&overlapTracking[neighborRow * 64 + neighborCol], 1); // TODO: this single instructions grows the execution time by 5x
                }
            }
        }

        if (!__any_sync(subgroupMask, neighborEnergy > 0)) {
            continue;
        }

        // printf("Expanding merged Pi0 at (%d, %d - Seed %d) with energy %d in direction %d which has energy %d\n", seedRow, seedCol, mergedPi0Indexes[pi0], seedEnergy, direction, pi0Energy);

        // add recently added cluster energy to original 3x3 neighbors, and add original cluster energy to recently added neighbors

        int expandedNeighborsEnergy = addedAsPi0Neighbor ? neighborEnergy : 0;
        for (int offset = 4; offset > 0; offset /= 2) {
            expandedNeighborsEnergy += __shfl_down_sync(subgroupMask, expandedNeighborsEnergy, offset);
        }

        expandedNeighborsEnergy = __shfl_sync(subgroupMask, expandedNeighborsEnergy, threadIdx.y % 4 * 8); // Broadcast to all threads

        // add original cluster energy + expanded neighbors energy to the neighborsTotClE array of the pi0 neighbors
        if (addedAsPi0Neighbor) {
            // printf("Digit at (%d, %d) added as neighbor to merged Pi0 at (%d, %d). Adding %d to %d\n", neighborRow, neighborCol, seedRow, seedCol, originalClusterEnergy + expandedNeighborsEnergy, neighborsTotClE[neighborRow * 64 + neighborCol]);
            atomicAdd(&neighborsTotClE[neighborRow * 64 + neighborCol], originalClusterEnergy + expandedNeighborsEnergy);
        }

        // add original neighbors and seed the energy of expanded neighbors
        neighborRow = seedRow + neighborOffsets[threadIdx.x][0];
        neighborCol = seedCol + neighborOffsets[threadIdx.x][1];

        if (neighborRow >= 0 && neighborRow < 58 && neighborCol >= 0 && neighborCol < 64) {
            if (caloValues[neighborRow][neighborCol] > 0) { // only if already has some energy, otherwise it was not used
                atomicAdd(&neighborsTotClE[neighborRow * 64 + neighborCol], expandedNeighborsEnergy);
            }
        }

        if (threadIdx.x == 0) {
            // printf("Adding expanded energy to seed at (%d, %d). Adding %d to %d\n", seedRow, seedCol, expandedNeighborsEnergy, neighborsTotClE[seedRow * 64 + seedCol]);
            neighborsTotClE[seedRow * 64 + seedCol] = originalClusterEnergy + expandedNeighborsEnergy;
        }
    }
}

__global__ void calculateWeightsV1(int* numSeeds, int* Seeds, int maxSeeds, int* adjList, int* adjListSizes, float* flatWeights, int8_t* isMergedPi0, int* expandedMergedPi0Neighbors, int* expandedMergedPi0NumNeighbors, float* expandedMergedPi0Weights, int* neighborsTotClE) {

    int eventIdx = blockIdx.x;

    int nSeeds = numSeeds[eventIdx];

    Seeds = Seeds + eventIdx * maxSeeds * 3;

    adjList = adjList + eventIdx * maxSeeds * 8 * 3;
    adjListSizes = adjListSizes + eventIdx * maxSeeds;
    flatWeights = flatWeights + eventIdx * maxSeeds * 8;

    isMergedPi0 = isMergedPi0 + eventIdx * maxSeeds;
    expandedMergedPi0Neighbors = expandedMergedPi0Neighbors + eventIdx * maxSeeds * 5 * 3;
    expandedMergedPi0NumNeighbors = expandedMergedPi0NumNeighbors + eventIdx * maxSeeds;
    expandedMergedPi0Weights = expandedMergedPi0Weights + eventIdx * maxSeeds * 5;

    neighborsTotClE = neighborsTotClE + eventIdx * 58 * 64;

    for (int seed = threadIdx.y; seed < nSeeds; seed += blockDim.y) {
        int clusterEnergy = neighborsTotClE[Seeds[seed * 3] * 64 + Seeds[seed * 3 + 1]];
        int neighborTotClEnergy;

        // weights of 3x3 window neighbors
        if (threadIdx.x < adjListSizes[seed]) {
            neighborTotClEnergy = neighborsTotClE[adjList[seed * 8 * 3 + threadIdx.x * 3] * 64 + adjList[seed * 8 * 3 + threadIdx.x * 3 + 1]];
            flatWeights[seed * 8 + threadIdx.x] = static_cast<float>(clusterEnergy) / neighborTotClEnergy;
        }

        // weights of merged Pi0 neighbors
        if (threadIdx.x < expandedMergedPi0NumNeighbors[seed]) {
            neighborTotClEnergy = neighborsTotClE[expandedMergedPi0Neighbors[seed * 5 * 3 + threadIdx.x * 3] * 64 + expandedMergedPi0Neighbors[seed * 5 * 3 + threadIdx.x * 3 + 1]];
            expandedMergedPi0Weights[seed * 5 + threadIdx.x] = static_cast<float>(clusterEnergy) / neighborTotClEnergy;
        }
    }
}
