// src/kernels/insertNodesAndEdges.cu
#include "InsertNodesAndEdges.h"

// __global__ void InsertNodesAndEdges(int* d_energy, int* d_id, int* d_neighbors, int numNodes, Graph* graph) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < numNodes) {
//         // For each energy and id
//         int energy = d_energy[idx];
//         int id = d_id[idx];

//         if (!graph->nodeExists(id)) {
//             // If the node is a local maxima, add to the graph
//             if (energy > 0) { 
//                 // Add node and neighbors
//                 for (int i = 0; i < d_neighbors[idx]; ++i) {
//                     int neighborId = d_neighbors[i];
//                     if (!graph->nodeExists(neighborId)) {
//                         graph->addEdge(neighborId, id, 1);
//                     }
//                 }
//             }
//         }
//     //     } else {
//     //         // If node is part of mergedPi0 (for simplicity, assume all are)
//     //         for (int i = 0; i < d_neighbors[idx]; ++i) {
//     //             int neighborId = d_neighbors[i];
//     //             if (!graph->nodeExists(neighborId)) {
//     //                 graph->addEdge(neighborId, id, 1);
//     //             }
//     //         }
//     //     }
//     }
// }