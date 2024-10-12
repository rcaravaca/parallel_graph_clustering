#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <iostream>
#include "utils.h"
#include "Digit.h"

#include <cmath>


/**
 * @brief Class representing a graph using an adjacency list.
 * This class is meant to be used on the CPU only.
 */
class Graph {
public:
    
    int numNodes;  ///< Number of nodes in the graph.

    /**
     * @brief Constructor: Initializes the graph with a given number of nodes.
     * @param nodes Number of nodes in the graph.
     */
    Graph();

    /**
     * @brief Adds an edge between two nodes with a weight.
     * @param source Source node of the edge.
     * @param destination Destination node of the edge.
     * @param weight Weight of the edge (default is 1).
     */
    void addEdge(const Digit& source, const Digit& destination);

    /**
     * @brief Adds a new node to the graph (only in host-side Graph object).
     * @param neighbors List of neighbors for the new node.
     */
    void addNode(const Digit& digit);

    /**
     * @brief Checks if a node exists in the graph.
     * @param id The node identifier to check.
     * @return True if the node exists, false otherwise.
     */
    bool nodeExists(const Digit& digit) const;

    /**
     * @brief Returns the node at the given index.
     * 
     * @param index The index of the node.
     * @return The node (Digit) at the given index.
     */
    const Digit& getNode(int index) const;

    /**
     * @brief Prints the adjacency list of the graph for debugging.
     */
    void printGraph() const;

    /**
     * @brief Get flatted adjacent list and size of list for CUDA
     */
    void flattenGraph(std::vector<int>& flatAdjList, std::vector<int>& adjListSizes, std::vector<int>& Nodes, int& numNodes) const;

    /**
     * @brief Rebuild the graph from flatten arrays.
     */
    void rebuildGraph(const std::vector<int>& flatAdjList, const std::vector<int>& adjListSizes, const std::vector<int>& Nodes, int numNodes);

    /**
     * @brief Print a summary of inserted nodes
     */
    void GraphSummary() const;

private:
    // std::vector<std::vector<std::pair<int, int>>> adjList;  ///< Adjacency list representing the graph (node, weight).
    std::vector<std::vector<Digit>> adjList;  ///< Adjacency list, where each vector contains a node and its neighbors.
};

#endif // GRAPH_H

