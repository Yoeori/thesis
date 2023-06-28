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

class ReducerRowBFS : public Vertex
{
public:

    InOut<unsigned int> dist;
    Output<bool> frontier;

    Input<Vector<unsigned int>> res;

    Input<unsigned int> iteration;

    auto compute() -> bool
    {
        *frontier = true;
        if (*dist != INT_MAX)
        {
            return true;
        }


        for (rptsize_t n = 0; n < res.size(); n+=1)
        {
            if (res[n])
            {
                *dist = iteration;
                *frontier = false;
                break;
            }
        }

        return true;
    }
};