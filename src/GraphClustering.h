#ifndef ADD_NODE_WITH_CUDA_H
#define ADD_NODE_WITH_CUDA_H

#include <vector>
#include "graph.h"

/**
 * @brief Inserts a set of Digits into the graph.
 * 
 * This function takes a vector of Digits and inserts them into the provided graph.
 * Only the Digits that meet certain criteria (e.g., energy greater than a threshold) are
 * added as nodes to the graph. The function ensures that the number of nodes 
 * does not exceed the specified maximum limit.
 * 
 * @param graph The reference to the Graph object where the nodes will be inserted.
 * @param maxNodes The maximum number of nodes allowed in the graph.
 * @param digits A vector of Digits containing the nodes to be processed and inserted into the graph.
 */
void GraphInsertion(Graph& graph, int maxNodes, const std::vector<Digit>& digits);

/**
 * @brief Inserts a set of Digits into the graph.
 * 
 * This function takes a vector of Digits and inserts them into the provided graph.
 * Only the Digits that meet certain criteria (e.g., energy greater than a threshold) are
 * added as nodes to the graph. The function ensures that the number of nodes 
 * does not exceed the specified maximum limit.
 * 
 * @param graph The reference to the Graph object where the nodes will be inserted.
 * @param maxNodes The maximum number of nodes allowed in the graph.
 * @param digits A vector of Digits containing the nodes to be processed and inserted into the graph.
 */
void GraphInsertionV2(Graph& graph, int maxNodes, const std::vector<Digit>& digits);

#endif // ADD_NODE_WITH_CUDA_H
