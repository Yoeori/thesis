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

class Normalize : public Vertex
{
public:
    InOut<Vector<unsigned int>> dist;
    InOut<Vector<float>> vec;

    Input<unsigned int> iteration;
    Output<bool> stop;

    auto compute() -> bool
    {
        *stop = true;
        for (rptsize_t i = 0; i < vec.size(); i += 1)
        {
            if (vec[i] >= 1.0)
            {
                if (dist[i] == UINT_MAX)
                {
                    dist[i] = *iteration;
                    vec[i] = 1.0;
                    *stop = false;
                }
                else
                {
                    vec[i] = 0.0;
                }
            }
        }

        return true;
    }
};