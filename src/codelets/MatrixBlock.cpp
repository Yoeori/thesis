#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>

using namespace poplar;

/*
(Please see these notes as what they are, random design rambling)
Two main parts codelets:
- MatrixBlockCalc
- Reducer 

We have the following data:
- Part of matrix (M)
    - idx_y_start / idx_y_end
    - idx_x_start / idx_x_end (or width / height)
    - matrix[y][x]
- Part of vector (x)
    - idx_start / idx_end (== idx_x_start / idx_x_end)
    - vector[x]

- These IDX's are not know to the Vertex, instead we assume that our vertex allocation was done correctly.

We save as a result:
M * x = y

Types:
- It should be preferred to use ints or another standard integer container, 
  however since we do not care for precision a.t.m and we don't know how big certain
  (vector) numbers can become we instead use floats, because _why not_

Compression:
We save a section of our (sparse) Matrix in three vectors:
- m: fixed array with all (non-zero) values of our 'rectangle'
- idx: contains the correspondin column for each value in m
- row: pointer to the first element of each row of our square

Next we have:
- vec: our uncompressed section of the vector (from col_start - col_end)
*/


class MatrixBlock : public Vertex {
public:

    // Our matrix is static
    Vector<float> m;
    Vector<size_t> idx; // Assume our indexes are 'normalized' for the block
    Vector<size_t> row_idx;

    InOut<Vector<float>> vec;
    InOut<Vector<float>> res;

    auto compute() -> bool {
        // Performs basic matrix * vector mult for block

        // Go by row
        for (auto i = 0; i < row_idx.size(); i++) {
            for (auto j = row_idx[i]; j < row_idx[j + 1]; j++) {
                res[i] += vec[idx[j]] * m[j];
            }
        }
        
        return true;
    }
};