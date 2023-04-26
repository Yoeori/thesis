#include <poplar/Vertex.hpp>
#include <poplar/Loops.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>
#include "print.h"

using namespace poplar;

class ReducerBFS : public MultiVertex
{
public:
    // We sum 0, n, n*2, n*3 ... to vector[0]
    // 1, n+1, n*2 + 1 ... to vector[1] etc.

    InOut<Vector<unsigned int>> dist;

    Input<Vector<bool>> res;
    Output<Vector<bool>> next_frontier;

    Input<unsigned int> iteration;

    int block_length;
    int res_block_length;
    int blocks;

    auto compute(unsigned workerId) -> bool
    {
        for (int i = workerId; i < block_length; i+= MultiVertex::numWorkers())
        {
            if (dist[i] != 0) {
                continue;
            }

            for (rptsize_t n = 0; n < blocks; n+=1)
            {
                if (res[n * res_block_length + i]) {
                    printf("iteration %d found %d\n", *iteration, i);
                    next_frontier[i] = true;
                    dist[i] = iteration;
                    goto cnt;
                }
            }

            next_frontier[i] = false;

            cnt:;
        }

        return true;
    }
};