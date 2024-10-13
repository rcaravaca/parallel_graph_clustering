#include <iostream>
#include <unordered_map>
#include <vector>
#include <rmm/device_uvector.hpp>
#include <raft/core/handle.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/algorithms.hpp>

// Definir una clase personalizada para cada nodo
class MyNode {
public:
    int id;
    std::string name;
    
    MyNode(int id, std::string name) : id(id), name(name) {}
    
    void print_info() const {
        std::cout << "Node ID: " << id << ", Name: " << name << std::endl;
    }
};

int main() {
    // Crear el "handle" de cuGraph
    raft::handle_t handle;

    // Definir las aristas del grafo (grafo dirigido)
    std::vector<int32_t> h_src = {0, 0, 1, 2, 2, 3};
    std::vector<int32_t> h_dst = {1, 2, 3, 3, 4, 4};
    std::vector<float> h_wgt = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};  // Pesos de las aristas

    // Número de vértices
    int32_t num_vertices = 5;

    // Asociar nodos a objetos en la CPU
    std::unordered_map<int, MyNode> node_objects;
    node_objects[0] = MyNode(0, "Node_A");
    node_objects[1] = MyNode(1, "Node_B");
    node_objects[2] = MyNode(2, "Node_C");
    node_objects[3] = MyNode(3, "Node_D");
    node_objects[4] = MyNode(4, "Node_E");

    // Imprimir información sobre los nodos
    for (const auto& pair : node_objects) {
        pair.second.print_info();
    }

    // Transferir datos a la GPU usando rmm::device_uvector
    rmm::device_uvector<int32_t> d_src(h_src.size(), handle.get_stream());
    rmm::device_uvector<int32_t> d_dst(h_dst.size(), handle.get_stream());
    rmm::device_uvector<float> d_wgt(h_wgt.size(), handle.get_stream());

    // Copiar los datos de la CPU a la GPU
    cudaMemcpy(d_src.data(), h_src.data(), sizeof(int32_t) * h_src.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst.data(), h_dst.data(), sizeof(int32_t) * h_dst.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wgt.data(), h_wgt.data(), sizeof(float) * h_wgt.size(), cudaMemcpyHostToDevice);

    // Crear el grafo dirigido
    cugraph::graph_t<int32_t, int32_t, float, true, false> graph(handle);
    graph.from_edgelist(d_src.data(), d_dst.data(), d_wgt.data(), num_vertices, h_src.size());

    // Ejecutar el algoritmo de PageRank como ejemplo
    rmm::device_uvector<float> d_pagerank(num_vertices, handle.get_stream());
    float alpha = 0.85f;
    float tolerance = 1e-6f;
    int max_iterations = 100;
    cugraph::pagerank(handle, graph.view(), d_pagerank.data(), nullptr, alpha, tolerance, max_iterations);

    // Obtener los resultados de la GPU y asociarlos con los objetos de nodos
    std::vector<float> h_pagerank(num_vertices);
    cudaMemcpy(h_pagerank.data(), d_pagerank.data(), sizeof(float) * num_vertices, cudaMemcpyDeviceToHost);

    // Mostrar el PageRank asociado a cada nodo-objeto
    std::cout << "PageRank Results: " << std::endl;
    for (int i = 0; i < num_vertices; ++i) {
        std::cout << "Node " << node_objects[i].name << " (" << node_objects[i].id << "): " << h_pagerank[i] << std::endl;
    }

    return 0;
}
