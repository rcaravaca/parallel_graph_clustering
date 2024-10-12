#include "GraphClustering.h"
#include "kernels/processGraph.h"

void GraphInsertion(Graph& graph, int maxNodes, const std::vector<Digit>& digits) {

    std::cout << "GraphInsertion: Starting the Graph Nodes Insertion..." << std::endl;

    std::vector<int> flatAdjList;
    std::vector<int> adjListSizes;
    std::vector<int> Nodes;
    int numNodes;

    // Aplanar el grafo en arrays
    graph.flattenGraph(flatAdjList, adjListSizes, Nodes, numNodes);

    // Crear arrays planos para los datos de los Digits
    int numDigits = digits.size();
    std::vector<int> rows(numDigits);
    std::vector<int> cols(numDigits);
    std::vector<int> energies(numDigits);

    for (int i = 0; i < numDigits; ++i) {
        rows[i] = digits[i].getRow();
        cols[i] = digits[i].getCol();
        energies[i] = digits[i].getEnergy();
    }

    // Calcular el tamaño necesario para Nodes y adjList
    int Nodesize = maxNodes * 3;       // 3 enteros por nodo
    int adjListSize = maxNodes * 9 * 3;  // Hasta 9 vecinos por nodo, cada uno con 3 enteros

    Nodes.resize(Nodesize);
    adjListSizes.resize(Nodesize);
    flatAdjList.resize(adjListSize);

    // Asignar memoria en el dispositivo
    int* d_adjList, *d_adjListSizes, *d_Nodes, *d_numNodes, *d_rows, *d_cols, *d_energies;
    cudaMalloc(&d_adjList, adjListSize * sizeof(int));  // Reservar suficiente espacio para adjList
    cudaMalloc(&d_adjListSizes, maxNodes * sizeof(int));  // Una entrada por cada nodo
    cudaMalloc(&d_Nodes, Nodesize * sizeof(int));  // Reservar suficiente espacio para los nodos
    cudaMalloc(&d_numNodes, sizeof(int));  // Un entero para el número de nodos
    cudaMalloc(&d_rows, rows.size() * sizeof(int));  // Filas de los Digits
    cudaMalloc(&d_cols, cols.size() * sizeof(int));  // Columnas de los Digits
    cudaMalloc(&d_energies, energies.size() * sizeof(int));  // Energías de los Digits

    // Copiar datos al dispositivo
    cudaMemcpy(d_adjList, flatAdjList.data(), adjListSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjListSizes, adjListSizes.data(), maxNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Nodes, Nodes.data(), Nodesize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numNodes, &numNodes, sizeof(int), cudaMemcpyHostToDevice);  // Inicializar numNodes a 0 en el dispositivo
    cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, energies.data(), energies.size() * sizeof(int), cudaMemcpyHostToDevice);


    // Configuración del kernel
    int blockSize = 256;
    int numBlocks = (numDigits + blockSize - 1) / blockSize;  // Calcular número de bloques

    // Lanzar el kernel para agregar los nodos
    addNodeToGraphCUDA<<<numBlocks, blockSize>>>(d_adjList, d_adjListSizes, d_Nodes, d_numNodes, maxNodes, d_rows, d_cols, d_energies, numDigits);

    // Sincronizar el dispositivo
    cudaDeviceSynchronize();

    // Copiar resultados de vuelta al host
    cudaMemcpy(flatAdjList.data(), d_adjList, adjListSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(adjListSizes.data(), d_adjListSizes, maxNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Nodes.data(), d_Nodes, Nodesize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&numNodes, d_numNodes, sizeof(int), cudaMemcpyDeviceToHost);  // Verificar que numNodes se actualice

    std::cout << "GraphInsertion: Count of added Nodes: " << numNodes << std::endl;

    // Reconstruir el grafo en el host
    graph.rebuildGraph(flatAdjList, adjListSizes, Nodes, numNodes);

    // Liberar memoria
    cudaFree(d_adjList);
    cudaFree(d_adjListSizes);
    cudaFree(d_Nodes);
    cudaFree(d_numNodes);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_energies);
    
    std::cout << "GraphInsertion: Done" << std::endl;

}
