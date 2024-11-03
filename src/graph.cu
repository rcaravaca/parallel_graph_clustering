#include "graph.h"

Graph::Graph(){};

void Graph::addEdge(const Digit& source, const Digit& destination, int weight) {
    // Search for the source node in adjList
    for (auto& nodeList : adjList) {
        if (nodeList.front().first.getID() == source.getID()) {
            nodeList.push_back({destination, weight});  // Add the destination node with the weight
            return;
        }
    }

    // If the source node is not found, print an error
    std::cerr << "Error: Source node not found." << std::endl;
}

void Graph::addNode(const Digit& newNode) {
    // Check if the node already exists in the graph (optional)
    for (const auto& nodeList : adjList) {
        if (nodeList.front().first.getID() == newNode.getID()) {
            std::cerr << "Error: Node with ID " << newNode.getID() << " row: " << newNode.getRow()<< " col: " << newNode.getCol() <<" already exists in the graph." << std::endl;
            return;
        }
    }

    // Add the new node to the graph with no neighbors (self-weight 0)
    adjList.push_back({{newNode, 0}});
}

bool Graph::nodeExists(const Digit& digit) const {
    for (const auto& nodeList : adjList) {
        if (nodeList.front().first.getID() == digit.getID()) {
            return true;
        }
    }
    return false;  // Node not founded
}

const Digit& Graph::getNode(int index) const {
    if (index >= 0 && index < adjList.size()) {
        return adjList[index].front().first;  // The first element in internal vector is the node
    } else {
        throw std::out_of_range("Index out of range.");
    }
}

void Graph::printGraph() const {
    int count = 0;  // Counter to keep track of the number of nodes printed

    // Loop through each node's adjacency list
    for (const auto& nodeList : adjList) {
        if (count >= 10) {
            break;  // Stop after printing 10 nodes
        }

        // The first element in the nodeList is the current node
        const Digit& node = nodeList.front().first;

        // Print the current node
        std::cout << "Node " << node.getID() << " (Row: " << node.getRow() 
                  << ", Col: " << node.getCol() << ", Energy: " << node.getEnergy() << ", Nº neighbors: " << nodeList.size() - 1 << "): ";

        // Print all its neighbors
        for (size_t i = 1; i < nodeList.size(); ++i) {
            const Digit& neighbor = nodeList[i].first;
            int weight = nodeList[i].second;  // Retrieve the weight for this neighbor

            // Print the neighbor and the edge weight
            std::cout << "Neighbor (" << neighbor.getRow() << ","<< neighbor.getCol() << "," << neighbor.getEnergy() << ") (Weight: " << weight << "), ";
        }

        std::cout << std::endl;  // End of this node's adjacency list

        count++;  // Increment the counter after printing a node
    }

    if (count == 0) {
        std::cout << "Graph is empty or has no nodes to print." << std::endl;
    }
}

void Graph::flattenGraph(std::vector<int>& flatAdjList, std::vector<int>& adjListSizes, std::vector<int>& Nodes, int& numNodes) const {

    numNodes = adjList.size();

    for (const auto& nodeList : adjList) {
        
        const Digit& node = nodeList.front().first;  // First element is the node itself
        
        Nodes.push_back(node.getRow());
        Nodes.push_back(node.getCol());
        Nodes.push_back(node.getEnergy());
        
        adjListSizes.push_back(flatAdjList.size()); 
        for (size_t i = 1; i < nodeList.size(); ++i) {
            const Digit& neighbor = nodeList[i].first;  // Neighbor and weight
            flatAdjList.push_back(neighbor.getRow());
            flatAdjList.push_back(neighbor.getCol());
            flatAdjList.push_back(neighbor.getEnergy());
            flatAdjList.push_back(nodeList[i].second);  // Include the weight in the same array
        }
    }
}

void Graph::rebuildGraph(const std::vector<int>& flatAdjList, const std::vector<int>& adjListSizes, const std::vector<int>& Nodes, int numNodes) {
    
    adjList.clear();  // clean the graph

    // Rebuild the graph
    for (int i = 0; i < numNodes; ++i) {
        
        // Every node has 3 values: row, col & energy
        int row = Nodes[i * 3]; 
        int col = Nodes[i * 3 + 1];
        int energy = Nodes[i * 3 + 2];

        // std::cout << "Nodo 0: Row=" << row << ", Col=" << col << ", Energy=" << energy << std::endl;

        // Make then main node and add to graph
        Digit node(row, col, energy);
        // adjList.push_back({{node, 0}});  // The first entry in the list, weight is 0 for the node itself
        addNode(node);

        int numNeighbors = adjListSizes[i];

        int offset = i * 8 * 4;  // Adjusted index for neighbors (8 neighbors, 4 values each)

        // add the neighbors
        for (int j = 0; j < numNeighbors; ++j) {
            int neighborRow = flatAdjList[offset];
            int neighborCol = flatAdjList[offset + 1];
            int neighborEnergy = flatAdjList[offset + 2];
            int weight = flatAdjList[offset + 3];  // Retrieve the weight

            Digit neighbor(neighborRow, neighborCol, neighborEnergy);
            // adjList.back().push_back({neighbor, weight});  // Add neighbor and weight
            addEdge(node, neighbor, weight);

            offset += 4;  // Adjust the offset for the next neighbor
        }
    }
}

void Graph::GraphSummary() const {

    if (adjList.empty()) {
        std::cout << "Graph is empty." << std::endl;
        return;
    }

    int numNodes = 0;
    int minEnergy = INT_MAX;
    int maxEnergy = INT_MIN;
    double sumEnergy = 0;
    double sumEnergySquared = 0; 

    for (const auto& nodeList : adjList) {
        if (!nodeList.empty()) {
            const Digit& node = nodeList.front().first;  

            int energy = node.getEnergy();
            numNodes++;

            if (energy < minEnergy) {
                minEnergy = energy;
            }
            if (energy > maxEnergy) {
                maxEnergy = energy;
            }

            sumEnergy += energy;
            sumEnergySquared += energy * energy;
        }
    }

    double meanEnergy = sumEnergy / numNodes;
    double variance = (sumEnergySquared / numNodes) - (meanEnergy * meanEnergy);
    double stdDeviation = sqrt(variance);

    std::cout << "\n##############################" << std::endl;
    std::cout << "Graph Summary:" << std::endl;
    std::cout << "Number of nodes: " << numNodes << std::endl;
    std::cout << "Energy values: " << std::endl;
    std::cout << "  Minimum: " << minEnergy << std::endl;
    std::cout << "  Maximum: " << maxEnergy << std::endl;
    std::cout << "  Average: " << meanEnergy << std::endl;
    std::cout << "  Standard deviation: " << stdDeviation << "\n##############################" << std::endl<< std::endl;
}


bool Graph::checkForDuplicateIDs() const {

    std::unordered_set<int> seenIDs;  // To track the IDs we have already encountered

    // Traverse the adjacency list
    for (const auto& nodeList : adjList) {
        // The first element in the nodeList is the current node
        const Digit& node = nodeList.front().first;

        // Check if this node's ID has already been seen
        if (seenIDs.find(node.getID()) != seenIDs.end()) {
            // Duplicate ID found, print the duplicate and return true
            std::cerr << "Duplicate ID found: Node " << node.getID() << std::endl;
            return true;
        }

        // If not seen, add this node's ID to the set
        seenIDs.insert(node.getID());
    }

    // If we reach here, no duplicate IDs were found
    std::cout << "No duplicate node IDs found." << std::endl;
    return false;
}
