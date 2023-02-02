#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>

using ::std::map;
using ::std::vector;
using ::std::string;
using ::std::optional;

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

using ::poplar::Device;
using ::poplar::Graph;
using ::poplar::Tensor;

using ::poplar::program::Program;
using ::poplar::program::Copy;

/*
Very simple test if example matrix works on the MatrixBlock codelet
*/
auto test_matrix_block_codelet(Device device, Graph graph) {
    // First define some sample values:

    // m = [ 1 -1
    //         -3 ]
    // v = [ 2  1 ]
    // m * v = [ 1 -3 ]

    // In our MatrixBlock format:
    auto m = {1.0, -1.0, -3.0};
    auto idx = {0, 1, 1};
    auto row_idx = { 0, 2, 3 };

    auto expect = { 1, -3 };

    // Flow:
    // 1. Copy
    // 2. Exec MatrixBlock
    // 3. Copy

    auto tensors = map<string, Tensor>{};
    tensors["m"] = graph.addVariable(poplar::FLOAT, {m.size()}, "m");
    tensors["idx"] = graph.addVariable(poplar::INT, {idx.size()}, "idx");
    tensors["row_idx"] = graph.addVariable(poplar::INT, {row_idx.size()}, "row_idx");

    auto cs = graph.addComputeSet("matrix_block");
    graph.addVertex(cs, "MatrixBlock", {
        {"m", tensors["m"]},
        {"idx", tensors["m"]},
        {"row_idx", tensors["m"]}
    });

    // Program;
    auto copym = Copy(toIpuStream, tensors["data"]);



}

auto buildComputeGraph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, const int numTiles) {
    // Add tensors
//     tensors["data"] = graph.addVariable(poplar::FLOAT, {0}, "data");
//     poputil::mapTensorLinearly(graph, tensors["data"]);


//     // Add programs and wire up data
//     const auto NumElemsPerTile = 0 / numTiles;
//     auto cs = graph.addComputeSet("loopBody");
//     for (auto tileNum = 0; tileNum < numTiles; tileNum++) {
//         const auto sliceEnd = std::min((tileNum + 1) * NumElemsPerTile, (int)   0);
//         const auto sliceStart = tileNum * NumElemsPerTile;

//         auto v = graph.addVertex(cs, "MatrixBlock", {
//                 {"data", tensors["data"].slice(sliceStart, sliceEnd)}
//         });
//         graph.setInitialValue(v["howMuchToAdd"], tileNum);
//         graph.setPerfEstimate(v, 100); // Ideally you'd get this as right as possible
//         graph.setTileMapping(v, tileNum);
//     }
//     auto executeIncrementVertex = Execute(cs);

//     auto mainProgram = Repeat(10, executeIncrementVertex, "repeat10x");
//     programs["main"] = mainProgram; // Program 0 will be the main program
// }

auto defineDataStreams(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs) {
    auto toIpuStream = graph.addHostToDeviceFIFO("TO_IPU", poplar::FLOAT, 0);
    auto fromIpuStream = graph.addDeviceToHostFIFO("FROM_IPU", poplar::FLOAT, 0);

    auto copyToIpuProgram = Copy(toIpuStream, tensors["data"]);
    auto copyToHostProgram = Copy(tensors["data"], fromIpuStream);

    programs["copy_to_ipu"] = copyToIpuProgram;
    programs["copy_to_host"] = copyToHostProgram;
}