#include "graph.h"

Graph::Graph(){};

void Graph::addEdge(const Digit& source, const Digit& destination) {
    // Buscar el nodo fuente en adjList
    for (auto& nodeList : adjList) {
        if (nodeList.front().getID() == source.getID()) {
            nodeList.push_back(destination);  // Add the destination node to neighbors list
            return;
        }
    }

    std::cerr << "Error: Source node not founded." << std::endl;
}

void Graph::addNode(const Digit& digit) {
    adjList.push_back(std::vector<Digit>{digit});  // the first elemnet is the node, without neighbors
}

bool Graph::nodeExists(const Digit& digit) const {
    for (const auto& nodeList : adjList) {
        if (nodeList.front().getID() == digit.getID()) {
            return true;
        }
    }
    return false;  // Node not founded
}

const Digit& Graph::getNode(int index) const {
    if (index >= 0 && index < adjList.size()) {
        return adjList[index].front();  // The first element in internal vector is the node
    } else {
        throw std::out_of_range("Index out of range.");
    }
}

void Graph::printGraph() const {
    for (const auto& nodeList : adjList) {
        const Digit& node = nodeList.front();
        std::cout << "Node " << node.getID() << " (Row=" << node.getRow() 
                  << ", Col=" << node.getCol() << ", Energy=" << node.getEnergy() << "): ";

        for (size_t i = 1; i < nodeList.size(); ++i) {
            const Digit& neighbor = nodeList[i];
            std::cout << "Digit(ID=" << neighbor.getID() << ", Row=" << neighbor.getRow()
                      << ", Col=" << neighbor.getCol() << ", Energy=" << neighbor.getEnergy() << ") ";
        }
        std::cout << std::endl;
    }
}

void Graph::flattenGraph(std::vector<int>& flatAdjList, std::vector<int>& adjListSizes, std::vector<int>& Nodes, int& numNodes) const {
    numNodes = adjList.size();
    for (const auto& nodeList : adjList) {
        // nodes.push_back(nodeList.front().getID());
        Nodes.push_back(nodeList.front().getRow());
        Nodes.push_back(nodeList.front().getCol());
        Nodes.push_back(nodeList.front().getEnergy());
        adjListSizes.push_back(flatAdjList.size()); 
        for (size_t i = 1; i < nodeList.size(); ++i) {
            // flatAdjList.push_back(nodeList[i].getID());
            flatAdjList.push_back(nodeList[i].getRow());
            flatAdjList.push_back(nodeList[i].getCol());
            flatAdjList.push_back(nodeList[i].getEnergy());
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
        // adjList.push_back({node});  // initual node withoput neighbors
        addNode(node);

        // Get the neighbor of node
        int startIdx = adjListSizes[i];  // Begining index of flatAdjList
        int endIdx = (i + 1 < adjListSizes.size()) ? adjListSizes[i + 1] : flatAdjList.size();  // last index
        // add the neighbors
        if (startIdx < endIdx) {
            for (int j = startIdx; j < endIdx; j += 3) {

                int neighborRow = flatAdjList[j];
                int neighborCol = flatAdjList[j + 1];
                int neighborEnergy = flatAdjList[j + 2];
                
                Digit neighbor(neighborRow, neighborCol, neighborEnergy);
                // adjList.back().push_back(neighbor);  // Añadir el vecino a la lista de adyacencia del nodo
                addEdge(node, neighbor);

            }
        }
    }
}

void Graph::GraphSummary() const {
    // Verificar si el grafo está vacío
    if (adjList.empty()) {
        std::cout << "Graph is empty." << std::endl;
        return;
    }

    // Variables para los cálculos
    int numNodes = 0;
    int minEnergy = INT_MAX;
    int maxEnergy = INT_MIN;
    double sumEnergy = 0;
    double sumEnergySquared = 0;  // Para calcular la desviación estándar

    // Recorrer todos los nodos y calcular los valores de energía
    for (const auto& nodeList : adjList) {
        if (!nodeList.empty()) {
            const Digit& node = nodeList.front();  // El nodo principal de la lista de adyacencia

            int energy = node.getEnergy();
            numNodes++;

            // Actualizar el valor mínimo y máximo de energía
            if (energy < minEnergy) {
                minEnergy = energy;
            }
            if (energy > maxEnergy) {
                maxEnergy = energy;
            }

            // Acumular la suma de las energías y la suma de las energías al cuadrado
            sumEnergy += energy;
            sumEnergySquared += energy * energy;
        }
    }

    // Cálculos finales
    double meanEnergy = sumEnergy / numNodes;
    double variance = (sumEnergySquared / numNodes) - (meanEnergy * meanEnergy);
    double stdDeviation = sqrt(variance);

    // Imprimir el resumen del grafo
    std::cout << "\n##############################" << std::endl;
    std::cout << "Graph Summary:" << std::endl;
    std::cout << "Number of nodes: " << numNodes << std::endl;
    std::cout << "Energy values: " << std::endl;
    std::cout << "  Minimum: " << minEnergy << std::endl;
    std::cout << "  Maximum: " << maxEnergy << std::endl;
    std::cout << "  Average: " << meanEnergy << std::endl;
    std::cout << "  Standard deviation: " << stdDeviation << "\n##############################" << std::endl<< std::endl;
}
