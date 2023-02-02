#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

using ::poplar::Device;
using ::poplar::Graph;


// Input: Matrix, Vector pair and IPU Graph / Device
// Defines the program flow for this experiment
auto exp_matrix_vector_calc() {

}

// Our codelets expect our data to be a certain format, instead of rebuilding our datastructure
// this function immediately reads MTX into this format
struct MatrixBlock {
    float *m;
    size_t *idx;
    size_t *row;
};


auto prepare_matrix(int width, int height) {


    return 0;
}

auto prepare_vector() {

}