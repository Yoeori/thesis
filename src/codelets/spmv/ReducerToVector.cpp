#include <poplar/Vertex.hpp>
#include <poplar/Loops.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>
// #include "print.h"

using namespace poplar;

class ReducerToVector : public MultiVertex
{
public:
    // We sum 0, n, n*2, n*3 ... to vector[0]
    // 1, n+1, n*2 + 1 ... to vector[1] etc.

    Vector<Input<Vector<float>>> res;
    Output<Vector<float>> vector;

    int block_length;
    int res_block_length;
    int blocks;

    auto compute(unsigned workerId) -> bool
    {
        for (int i = workerId; i < block_length; i+= MultiVertex::numWorkers())
        {
            auto sum = 0;
            for (rptsize_t n = 0; n < blocks; n+=1)
            {
                sum += res[n][i];
            }
            vector[i] = sum;
        }

        return true;
    }
};