#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>

using namespace poplar;

class ReducerToVector : public Vertex
{
public:

    // We sum 0, n, n*2, n*3 ... to vector[0]
    // 1, n+1, n*2 + 1 ... to vector[1] etc.

    Input<Vector<float>> res;
    Output<Vector<float>> vector;

    int block_length;
    int blocks;

    auto compute() -> bool
    {
        for (int i = 0; i < block_length; i++) {
            vector[i] = 0;
            for (int n = 0; n < blocks; n++) {
                vector[i] += res[n * block_length + i];
            }
        }

        return true;
    }
};