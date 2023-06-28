#include <poplar/Vertex.hpp>
#include <poplar/Loops.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>
#include <limits.h>

using namespace poplar;

class ReducerRowBFSMulti : public MultiVertex
{
public:

    InOut<Vector<unsigned int>> dist;
    Output<Vector<bool>> frontier;

    // Input<Vector<unsigned int>> res;
    Input<VectorList<unsigned int, VectorListLayout::DELTANELEMENTS>> res;

    Input<unsigned int> iteration;

    auto compute(unsigned workerId) -> bool
    {
        for (size_t row = workerId; row < res.size(); row+=MultiVertex::numWorkers())
        {
            frontier[row] = true;

            if (dist[row] != INT_MAX)
            {
                continue;
            }

            for (rptsize_t n = 0; n < res.size(); n+=1) // innerloop
            {
                if (res[row][n] == 1)
                {
                    dist[row] = iteration;
                    frontier[row] = false;
                    break; // breaks innerloop
                }
            }
        }

        return true;
    }
};