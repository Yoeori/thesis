#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>

using namespace poplar;

class Reducer : public Vertex {
public:
    InOut <Vector<float>> result;
    InOut <Vector<float>> data;

    int row_space;
    int blocks;

    auto compute() -> bool {

        for (auto block = 0; block < blocks; block++) {
            for (auto i = 0; i < row_space; i++) {
                result[i] += data[block * row_space + i];
            }
        }

        return true;
    }
};