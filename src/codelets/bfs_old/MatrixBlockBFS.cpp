#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>

using namespace poplar;

class MatrixBlockBFS : public MultiVertex
{
public:
    // Data structure:
    // m[i] = M_(E_t where row_idx[t] >= i and row_idx[t + 1] < i ==> t, idx[i])
    Input<Vector<int>> idx;
    Input<Vector<int>> row_idx;

    Input<Vector<bool>> frontier;
    Output<Vector<unsigned int>> res;

    Input<unsigned int> iteration;

    auto compute(unsigned workerId) -> bool
    {
        // Performs basic matrix * vector mult for block
        // Go by row
        for (auto i = workerId; i < row_idx.size() - 1; i+= MultiVertex::numWorkers())
        {
            res[i] = 0;
            for (auto j = row_idx[i]; j < row_idx[i + 1]; j++)
            {
                if (!frontier[idx[j]]) 
                {
                    res[i] = iteration;
                    goto cnt;
                }
            }

            cnt:;
        }

        return true;
    }
};