
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

        }
    }
}



