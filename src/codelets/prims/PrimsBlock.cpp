#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>
#include <float.h>
#include <limits.h>
#include <print.h>

using namespace poplar;

class PrimsBlock : public MultiVertex
{
public:
    // Our matrix in CSC form
    Input<Vector<int>> weights;
    Input<Vector<unsigned>> rows;
    Input<Vector<unsigned>> columns;

    // The current column over which we are updating dist
    // TODO row offset, setting dist to infinity
    Input<unsigned> current;
    unsigned row_offset;

    InOut<Vector<int>> dist;
    InOut<Vector<unsigned>> dist_prev;

    auto compute(unsigned workerId) -> bool
    {
        if (workerId == 0 && current >= row_offset && current - row_offset < dist.size())
        {
            dist[current - row_offset] = INT_MAX;
            dist_prev[current - row_offset] = UINT_MAX;
        }

        // MultiVertex safety: dist and dist_prev are aligned per row (32-bit word), each column only contains a row once max.
        for (size_t i = columns[current] + workerId; i < columns[current + 1]; i+= MultiVertex::numWorkers()) {
            unsigned row = rows[i];

            if (weights[i] < dist[row] && dist_prev[row] != UINT_MAX) {
                // printf("Found: %d at dist %d\n", row + row_offset, weights[i]);
                dist[row] = weights[i];
                dist_prev[row] = *current;
            }
        }
        
        return true;
    }
};