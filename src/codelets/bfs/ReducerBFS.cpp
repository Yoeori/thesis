#include <poplar/Vertex.hpp>
#include <poplar/Loops.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>

using namespace poplar;

class ReducerBFS : public MultiVertex
{
public:
    // We sum 0, n, n*2, n*3 ... to vector[0]
    // 1, n+1, n*2 + 1 ... to vector[1] etc.

    InOut<Vector<unsigned int>> dist;

    Vector<Input<Vector<unsigned int>>> res;
    Output<Vector<bool>> frontier;

    Input<unsigned int> iteration;
    Output<bool> stop;

    int block_length;
    int blocks;

    auto compute(unsigned workerId) -> bool
    {
        if (workerId == 0) {
            *stop = true;
        }

        for (int i = workerId; i < block_length; i+= MultiVertex::numWorkers())
        {
            if (dist[i] != 0) {
                continue;
            }

            for (rptsize_t n = 0; n < blocks; n+=1)
            {
                if (res[n][i] == 1) {
                    frontier[i] = true; 
                    *stop = false; // Race condition safety: control flow will not be reached before line 35 is executed.
                    dist[i] = iteration;
                    goto cnt;
                }
            }

            frontier[i] = false;

            cnt:;
        }

        return true;
    }
};