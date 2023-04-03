#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>

using namespace poplar;

class Reducer : public Vertex
{
public:
    InOut<Vector<float>> vs;
    int block_length;

    auto compute() -> bool
    {
        int last_block = (vs.size() / block_length - 1) * block_length;
        for (int i = 0; i < vs.size() / block_length; i++)
        {
            for (int n = 0; n < block_length; n++)
            {
                vs[last_block + n] += vs[block_length * i + n];
            }
        }

        return true;
    }
};