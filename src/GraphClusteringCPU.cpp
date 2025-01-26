#include "GraphClusteringCPU.h"
#include <algorithm>
#include <numeric>

void graphInsertion(Graph& graph, int maxNodes, std::vector<Digit>& digits) {
    
    std::sort(digits.begin(), digits.end(), [](const Digit& a, const Digit& b) {
        return a.getEnergy() > b.getEnergy();
    });

    // std::vector<int> mergedPi0;
    // mergedPi0.reserve( 1024 );

    std::map<int, int> outEdges;


    std::vector<std::vector<int>> matrix(58, std::vector<int>(64, 0));
    for (const Digit& digit : digits) {
        matrix[digit.getRow()][digit.getCol()] = digit.getEnergy();
    }

    int neighborOffsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, // Top-left, Top, Top-right
        {0, -1},          {0, 1},  // Left,       Right
        {1, -1}, {1, 0}, {1, 1}    // Bottom-left, Bottom, Bottom-right
    };
    // Insert nodes into the graph
    for (const Digit& digit : digits) {
        if (digit.getEnergy() <= 50) {
            continue;
        }
        bool isLocalMax = true;
        for (int i = 0; i < 8; i++) {
            int row = digit.getRow() + neighborOffsets[i][0];
            int col = digit.getCol() + neighborOffsets[i][1];
            if (row >= 0 && row < 58 && col >= 0 && col < 64) {
                if (matrix[row][col] > digit.getEnergy()) {
                    isLocalMax = false;
                    break;
                }
            }
        }

        if (isLocalMax) {
            // Add the node to the graph
            graph.addNode(digit);
            
            for (int i = 0; i < 8; i++) {
                int row = digit.getRow() + neighborOffsets[i][0];
                int col = digit.getCol() + neighborOffsets[i][1];
                if (row >= 0 && row < 58 && col >= 0 && col < 64) {
                    if (matrix[row][col] > 0) {
                        graph.addEdge(digit, Digit(row, col, matrix[row][col]), 1);
                    }
                }
            }
        }
    }
}

void GraphInsertionNEventsCPU(
    std::vector<Graph>& graphs,
    const int maxSeeds,
    const std::vector<Digit>& digits,
    const std::vector<int>& digitsOffsets,
    const std::vector<int>& numDigits) {

    // Assume number of events == number of graphs == size of digitsOffsets == size of numDigits

    if (graphs.size() != digitsOffsets.size() || graphs.size() != numDigits.size()) {
        std::cerr << "GraphInsertionNEventsCPU: Error: Number of graphs, digitsOffsets, and numDigits do not match." << std::endl;
        return;
    }

    int numEvents = graphs.size();

    for (int i = 0; i < numEvents; i++) {
        int start = digitsOffsets[i];
        int end = start + numDigits[i];

        std::vector<Digit> eventDigits(digits.begin() + start, digits.begin() + end);

        graphInsertion(graphs[i], maxSeeds, eventDigits);
    }

}

void DFSUtil( int v, std::vector<bool>& visited, std::vector<std::vector<int>>& conComp,
                         int index, std::vector<std::vector<int>> m_adj_u) {
      visited[v] = true; // Mark the current node as visited and print it
      conComp[index].emplace_back( v );
      // Recur for all the vertices adjacent to this vertex
      for ( auto const& i : m_adj_u[v] ) {
        if ( !visited[i] ) { DFSUtil( i, visited, conComp, index, m_adj_u); }
      }
    }

auto connectedComponents(size_t m_size, std::vector<std::vector<int>> m_adj_u) {
    std::vector<std::vector<int>> conComp( m_size );
    std::vector<bool>      visited( m_size ); // Mark all the vertices as not visited
    for ( size_t v = 0; v < m_size; v++ ) {
        if ( !visited[v] && !m_adj_u[v].empty() ) {
            printf("digit at (%ld %ld) visited? %s has m_adj_size: %ld\n", v / 66, v % 66, visited[v] ? "true" : "false", m_adj_u[v].size());
            DFSUtil( v, visited, conComp, v , m_adj_u); // Look all reachable vertices from v
        }
    }
    return conComp;
}


void GraphInsertionCPUWithPi0AndWeights(
    const std::vector<Digit>& digits,
    const std::vector<int>& digitsOffsets,
    const std::vector<int>& numDigits) {

    size_t           m_size{6016};
    std::vector<std::vector<int>> m_adj_u(m_size);
    std::vector<std::map<int, float>> m_adj_weighted(m_size);
    std::vector<std::map<int, float>> m_adj_weighted_pred(m_size);

    int start = digitsOffsets[0];
    int end = start + numDigits[0];

    std::vector<Digit> eventDigits(digits.begin() + start, digits.begin() + end);

    std::sort(eventDigits.begin(), eventDigits.end(), [](const Digit& a, const Digit& b) {
        return a.getEnergy() > b.getEnergy();
    });

    printf("Event digits size: %ld\n", eventDigits.size());
    printf("most energetic digit: %d\n", eventDigits[0].getEnergy());
    printf("least energetic digit: %d\n", eventDigits[eventDigits.size() - 1].getEnergy());

    std::vector<std::vector<int>> matrix(58, std::vector<int>(66, 0));
    for (const Digit& digit : digits) {
        matrix[digit.getRow()][digit.getCol()] = digit.getEnergy();
    }

    int neighborOffsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, // Top-left, Top, Top-right
        {0, -1},          {0, 1},  // Left,       Right
        {1, -1}, {1, 0}, {1, 1}    // Bottom-left, Bottom, Bottom-right
    };

    std::vector<int> mergedPi0;
    mergedPi0.reserve( 1024 );

    for (const Digit& digit : eventDigits) {
        if (digit.getEnergy() <= 50) {
            continue;
        }

        if (m_adj_weighted[digit.getID()].size() > 0) {
            // printf("Searching for digit with ID %d in mergedPi0 (row %d, col %d, energy %d)\n", digit.getID(), digit.getRow(), digit.getCol(), digit.getEnergy());
            if (std::find(mergedPi0.begin(), mergedPi0.end(), digit.getID()) != mergedPi0.end()) {
                int seed = m_adj_weighted[digit.getID()].begin()->first;

                for (int i = 0; i < 8; i++) {
                    int row = digit.getRow() + neighborOffsets[i][0];
                    int col = digit.getCol() + neighborOffsets[i][1];
                    if (row >= 0 && row < 58 && col >= 0 && col < 66) {
                        if (matrix[row][col] > 0) {
                            if (digit.getEnergy() > matrix[row][col] && m_adj_weighted[row * 66 + col].size() == 0) {
                                m_adj_u[row * 66 + col].push_back(seed);
                                m_adj_u[seed].push_back(row * 66 + col);
                                m_adj_weighted[row * 66 + col][seed] = 1;
                                m_adj_weighted_pred[seed][row * 66 + col] = 1;
                            }
                        }
                    }
                }
            } 
        }


        bool isLocalMax = true;
        
        for (int i = 0; i < 8; i++) {
            int row = digit.getRow() + neighborOffsets[i][0];
            int col = digit.getCol() + neighborOffsets[i][1];
            if (row >= 0 && row < 58 && col >= 0 && col < 66) {
                if (matrix[row][col] > digit.getEnergy()) {
                    isLocalMax = false;
                    break;
                }
            }
        }

        if (isLocalMax) {
            for (int i = 0; i < 8; i++)  {
                int row = digit.getRow() + neighborOffsets[i][0];
                int col = digit.getCol() + neighborOffsets[i][1];
                if (row >= 0 && row < 58 && col >= 0 && col < 66) {
                    if (matrix[row][col] > 0) {

                        m_adj_u[digit.getID()].push_back(row * 66 + col);
                        m_adj_u[row * 66 + col].push_back(digit.getID());
                        m_adj_weighted[row * 66 + col][digit.getID()] = 1;
                        m_adj_weighted_pred[digit.getID()][row * 66 + col] = 1;

                        if (digit.getEnergy() == 206) {
                            printf("Found digit with energy 206 at (%d, %d) is a local max: %d\n", digit.getRow(), digit.getCol(), isLocalMax);
                            printf("%ld\n", m_adj_weighted_pred[digit.getID()].size());
                        }

                        if (digit.getEnergy() > 1000 && matrix[row][col] * 100 / digit.getEnergy() > 25) {
                            mergedPi0.push_back(digit.getID());
                            mergedPi0.push_back(row * 66 + col);
                        }
                    }
                }
            }
        }

    }

    auto conComp = connectedComponents(m_size, m_adj_u);

    int count = 0;
    int biggestComp = 0;
    for (size_t c = 0; c < m_size; c++) {
        if (conComp[c].size() > 1) {
            count++;
            if (conComp[c].size() > biggestComp) {
                biggestComp = conComp[c].size();
                printf("Biggest component size includes digit at (%ld, %ld) with energy %d\n", c / 66, c % 66, matrix[c / 66][c % 66]);
            }
        }
    }

    printf("Connected components size: %d\n", count);
    printf("Biggest component size: %d\n", biggestComp);

    std::map<int, float> seedToClEnergy;

    // for (auto connectedComponent : conComp) {
    //     for (auto node: connectedComponent) {
    //         const auto& out_edges = m_adj_weighted[node];
    //         if (out_edges.size() < 2) {
    //             continue;
    //         }
    //         auto totalEnergy{0.f};
    //         for ( auto edge : out_edges ) { // iter over out edges -> seeds this neighbor is a neighbor of
    //         const auto seedId = edge.first; // edge is a pair (seed, weight) .first is the seed
    //         if ( const auto energy_e = seedToClEnergy.find( seedId ); energy_e != seedToClEnergy.end() ) { // check if already calculated
    //             totalEnergy += energy_e->second;
    //             continue;
    //         }
    //         int seedRow = seedId / 66;
    //         int seedCol = seedId % 66;
    //         int seedEnergy = matrix[seedRow][seedCol];
    //         const auto& cells = m_adj_weighted_pred[seedId]; // get in edges of seeds -> neighbors pointing to this seed
    //         float       energy =
    //             std::accumulate( cells.begin(), cells.end(), seedEnergy > 0 ? seedEnergy / out_edges.size() : 0.f,
    //                             [&]( float energy, const auto& te ) {
    //                                 auto cell = digits.find( Detector::Calo::DenseIndex::details::toCellID( te.first ) );
    //                                 return cell ? energy + cell->energy() : energy;
    //                             } );
    //         totalEnergy += energy;  
    //         [[maybe_unused]] auto r = seedToClEnergy.emplace( seedId, energy );
    //         }
    //         for ( auto out_edge : out_edges ) {
    //         const auto seed            = out_edge.first;
    //         auto       weight          = seedToClEnergy.at( seed ) / totalEnergy;
    //         m_adj_weighted[node][seed]            = weight;
    //         m_adj_weighted_pred[seed][node]       = weight;
    //         }
    //     }
    // }

}