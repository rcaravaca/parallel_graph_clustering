#ifndef ADD_NODE_WITH_CPU_H
#define ADD_NODE_WITH_CPU_H

#include <vector>
#include "graph.h"

void GraphInsertionNEventsCPU(
    std::vector<Graph>& graphs,
    const int maxSeeds,
    const std::vector<Digit>& digits,
    const std::vector<int>& digitsOffsets,
    const std::vector<int>& numDigits);

#endif // ADD_NODE_WITH_CPU_H