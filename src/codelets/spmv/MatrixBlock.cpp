#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>
// #include "print.h"

using namespace poplar;

class MatrixBlock : public MultiVertex
{
public:
    // Data structure:
    // m[i] = M_(E_t where row_idx[t] >= i and row_idx[t + 1] < i ==> t, idx[i])
    Input<Vector<float>> matrix;
    Input<Vector<int>> idx; // TODO: Assume our indexes are 'normalized' for the block
    Input<Vector<int>> row_idx;

    Input<Vector<float>> vec;
    Output<Vector<float>> res;

    auto compute(unsigned workerId) -> bool
    {
        // Performs basic matrix * vector mult for block
        // Go by row
        for (auto i = workerId; i < row_idx.size() - 1; i+= MultiVertex::numWorkers())
        {
            float sum = 0.0;
            for (auto j = row_idx[i]; j < row_idx[i + 1]; j++)
            {
                sum += vec[idx[j]] * matrix[j];
            }
            res[i] = sum;
        }

        return true;
    }
};