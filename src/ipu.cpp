#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>
#include <cmath>

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

#include "matrix.cpp"

using ::std::map;
using ::std::optional;
using ::std::string;
using ::std::vector;

using ::poplar::Device;
using ::poplar::DeviceManager;
using ::poplar::Engine;
using ::poplar::FLOAT;
using ::poplar::Graph;
using ::poplar::INT;
using ::poplar::OptionFlags;
using ::poplar::TargetType;
using ::poplar::Tensor;
using ::poplar::program::Copy;
using ::poplar::program::Execute;
using ::poplar::program::Program;
using ::poplar::program::Repeat;
using ::poplar::program::Sequence;

optional<Device> getIpuDevice(const unsigned int numIpus = 1)
{
    DeviceManager manager = DeviceManager::createDeviceManager();
    optional<Device> device = std::nullopt;
    for (auto &d : manager.getDevices(TargetType::IPU, numIpus))
    {
        std::cout << "Trying to attach to IPU " << d.getId();
        if (d.attach())
        {
            std::cout << " - attached" << std::endl;
            device = {std::move(d)};
            break;
        }
        else
        {
            std::cout << std::endl
                      << "Error attaching to device" << std::endl;
        }
    }
    return device;
}

optional<Device> getIpuModel(const unsigned int numIpus = 1, const unsigned int tilesPerIpu = 10)
{
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs = numIpus;
    ipuModel.tilesPerIPU = tilesPerIpu;
    return ipuModel.createDevice();
}

/*
We define the following programs:
1. Copy Matrix from host top IPU
2. Copy Vector from host to IPU
3. Execute SpMV
4. Reduce and spread Vector result
4. Copy Vector back to host
*/

auto create_graph_add_codelets(const Device &device) -> Graph
{
    auto graph = poplar::Graph(device.getTarget());

    // Add our own codelets
    graph.addCodelets({"codelets/MatrixBlock.cpp", "codelets/Reducer.cpp", "codelets/ReducerToVector.cpp"}, "-O3 -I codelets");
    popops::addCodelets(graph);

    return graph;
}

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct IPUMatrix
{
public:
    IPUMatrix(vector<int> offsets, vector<T> matrix, vector<int> idx, vector<int> row_idx, int blocks, int block_height, int m, int n) : offsets(offsets), matrix(matrix), idx(idx), row_idx(row_idx), blocks(blocks), block_height(block_height), m(m), n(n) {}

    vector<int> offsets;

    // actuall data
    vector<T> matrix;
    vector<int> idx; // better indexing type? size_t?
    vector<int> row_idx;

    // matrix data
    int blocks;
    int block_height;
    int m;
    int n;

};

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
IPUMatrix<T> prepare_data(matrix::Matrix<T> matrix, const int num_tiles)
{
    // assumptions at this point in the code: matrix is shuffled (values are normally divided)
    // TODO: prepareData currently takes O(n*m), can be done in O(nz) for SparseMatrix type

    // First we calculate how many blocks we have available. We need x tiles for summation and x^2 blocks for the SpMV
    // In general this _should_ make it possible to execute SpMV on the same matrix twice with differents vectors.
    const auto blocks = (int)std::floor((-1.0 + std::sqrt(1 + 4 * num_tiles)) / 2.0); // For a standard IPU 37*37
    const auto block_size_col = std::max(matrix.cols() / blocks + (matrix.cols() % blocks != 0), 1);
    const auto block_size_row = std::max(matrix.rows() / blocks + (matrix.rows() % blocks != 0), 1);

    vector<T> ipu_matrix(matrix.nonzeroes());
    vector<int> idx(matrix.nonzeroes());

    // Could be more compact (the last row might need less space), but this complicates location calculations _a lot_
    vector<int> row_idx(blocks * blocks * (block_size_row + 1));

    // Next we perform summation over the matrix to find exact length for each block
    // TODO: we should/could log normality of sparse matrix
    // In general the row_idx length is the same for each block, with the expection of the last row of blocks. (being ceil(matrix.m / blocks))

    // This will give the offsets for matrix and idx
    vector<int> offsets(blocks * blocks + 1);
    offsets[0] = 0;

    for (auto y = 0; y < blocks; y++)
    {
        for (auto x = 0; x < blocks; x++)
        {
            offsets[y * blocks + x + 1] = offsets[y * blocks + x];

            // Search block for non-zero
            for (auto mi = block_size_row * y; mi < std::min(block_size_row * (y + 1), matrix.rows()); mi++)
            {
                // Record row offsets
                row_idx[(y * blocks + x) * (block_size_row + 1) + mi - block_size_row * y] = offsets[y * blocks + x + 1] - offsets[y * blocks + x];


                for (auto mj = block_size_col * x; mj < std::min(block_size_col * (x + 1), matrix.cols()); mj++)
                {
                    if (matrix.get(mi, mj) != 0)
                    {
                        ipu_matrix[offsets[y * blocks + x + 1]] = matrix.get(mj, mi);
                        idx[offsets[y * blocks + x + 1]] = mj;// - block_size_col * x;

                        offsets[y * blocks + x + 1] += 1;
                    }
                }
            }

            row_idx[(y * blocks + x) * (block_size_row + 1) + std::min(block_size_row * (y + 1), matrix.rows()) - block_size_row * y] = offsets[y * blocks + x + 1] - offsets[y * blocks + x];
        }
    }

    // Final value should be nz (sum of every block)
    assert(offsets[offsets.size() - 1] == matrix.nonzeroes());

    return IPUMatrix(offsets, ipu_matrix, idx, row_idx, blocks, block_size_row, matrix.rows(), matrix.cols());
}

// template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
// void prepareData(const matrix::SparseMatrix<T> matrix, const int num_tiles)
// {

// }

void build_compute_graph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, const int num_tiles, IPUMatrix<float> &ipu_matrix)
{

    // Static Matrix data
    tensors["matrix"] = graph.addVariable(FLOAT, {ipu_matrix.matrix.size()}, "matrix");
    poputil::mapTensorLinearly(graph, tensors["matrix"]); // TODO: is it usefull/necessary to apply a simple mapping?

    tensors["idx"] = graph.addVariable(INT, {ipu_matrix.idx.size()}, "idx");
    poputil::mapTensorLinearly(graph, tensors["idx"]);

    tensors["row_idx"] = graph.addVariable(INT, {ipu_matrix.row_idx.size()}, "row_idx");
    poputil::mapTensorLinearly(graph, tensors["row_idx"]);

    // Input/Output vector
    tensors["vector"] = graph.addVariable(FLOAT, {(unsigned int)ipu_matrix.n}, "vector");
    poputil::mapTensorLinearly(graph, tensors["vector"]);

    tensors["res"] = graph.addVariable(FLOAT, {(unsigned int)ipu_matrix.blocks * ipu_matrix.blocks * ipu_matrix.block_height}, "result");
    poputil::mapTensorLinearly(graph, tensors["res"]);

    // We build the compute set for the MatrixBlock codelet
    auto spmv_cs = graph.addComputeSet("spmv");

    for (auto y = 0; y < ipu_matrix.blocks; y++)
    {
        for (auto x = 0; x < ipu_matrix.blocks; x++)
        {
            auto block_id = y * ipu_matrix.blocks + x;
            auto v = graph.addVertex(spmv_cs, "MatrixBlock", {
                {"matrix", tensors["matrix"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1])},
                {"idx", tensors["idx"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1])},
                {"row_idx", tensors["row_idx"].slice((y * ipu_matrix.blocks + x) * (ipu_matrix.block_height + 1), (y * ipu_matrix.blocks + x + 1) * (ipu_matrix.block_height + 1))},
                {"vec", tensors["vector"]},
                {"res", tensors["res"].slice(block_id * ipu_matrix.block_height, (block_id + 1) * ipu_matrix.block_height)}
            });

            // TODO need to be calculated;
            graph.setPerfEstimate(v, 100);
            graph.setTileMapping(v, block_id);
        }
    }

    programs["spmv"] = Execute(spmv_cs);

    // We build the compute set for addition
    auto reducer_cs = graph.addComputeSet("reduce");

    for (auto y = 0; y < ipu_matrix.blocks; y++) {
        
        auto v = graph.addVertex(reducer_cs, "ReducerToVector", {
            {"res", tensors["res"].slice(y * ipu_matrix.blocks * ipu_matrix.block_height, (y + 1) * ipu_matrix.blocks * ipu_matrix.block_height)},
            {"vector", tensors["vector"].slice(y * ipu_matrix.block_height, (y + 1) * ipu_matrix.block_height)}
        });

        if (y == ipu_matrix.blocks - 1) {
            graph.setInitialValue(v["block_length"], std::max(0, ipu_matrix.m - ipu_matrix.block_height * ipu_matrix.blocks));
        } else {
            graph.setInitialValue(v["block_length"], ipu_matrix.block_height);
        }
        graph.setInitialValue(v["blocks"], ipu_matrix.blocks);

        graph.setPerfEstimate(v, 100);
        graph.setTileMapping(v, ipu_matrix.blocks * ipu_matrix.blocks + y);
    }

    programs["reduce"] = Execute(reducer_cs);
}

auto build_data_streams(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, IPUMatrix<float> &ipu_matrix)
{
    auto toipu_matrix = graph.addHostToDeviceFIFO("toipu_matrix", FLOAT, ipu_matrix.matrix.size());
    auto toipu_idx = graph.addHostToDeviceFIFO("toipu_idx", INT, ipu_matrix.idx.size());
    auto toipu_row_idx = graph.addHostToDeviceFIFO("toipu_row_idx", INT, ipu_matrix.row_idx.size());
    auto toipu_vec = graph.addHostToDeviceFIFO("toipu_vec", FLOAT, ipu_matrix.n);

    auto fromipu_vec = graph.addDeviceToHostFIFO("fromipu_vec", FLOAT, ipu_matrix.n);

    auto copyto_matrix = Copy(toipu_matrix, tensors["matrix"]);
    auto copyto_idx = Copy(toipu_idx, tensors["idx"]);
    auto copyto_row_idx = Copy(toipu_row_idx, tensors["row_idx"]);
    auto copyto_vec = Copy(toipu_vec, tensors["vector"]);

    auto copyhost_vec = Copy(tensors["vector"], fromipu_vec);

    programs["copy_to_ipu_matrix"] = Sequence{copyto_matrix, copyto_idx, copyto_row_idx};
    programs["copy_to_ipu_vec"] = copyto_vec;

    programs["copy_to_host"] = copyhost_vec;
}

auto serialize_graph(const Graph &graph)
{
    std::ofstream graphSerOfs;
    graphSerOfs.open("serialized_graph.capnp", std::ofstream::out | std::ofstream::trunc);

    graph.serialize(graphSerOfs, poplar::SerializationFormat::Binary);
    graphSerOfs.close();
}

auto runIPU()
{
    // std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
    // auto device = getIpuDevice(1);
    // // auto device = getIpuModel(1, 10);
    // if (!device.has_value())
    // {
    //     std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
    //     return EXIT_FAILURE;
    // }

    // std::cout << "STEP 2: Create graph and compile codelets" << std::endl;
    // auto graph = createGraphAndAddCodelets(device.value());

    // std::cout << "STEP 3: Building the compute graph" << std::endl;
    // auto tensors = map<string, Tensor>{};
    // auto programs = map<string, Program>{};
    // buildComputeGraph(graph, tensors, programs, device->getTarget().getNumTiles());

    // std::cout << "STEP 4: Define data streams" << std::endl;
    // defineDataStreams(graph, tensors, programs);

    // std::cout << "STEP 5: Create engine and compile graph" << std::endl;
    // auto ENGINE_OPTIONS = OptionFlags{
    //     {"target.saveArchive", "archive.a"},
    //     {"debug.instrument", "true"},
    //     {"debug.instrumentCompute", "true"},
    //     {"debug.loweredVarDumpFile", "vars.capnp"},
    //     {"debug.instrumentControlFlow", "true"},
    //     {"debug.computeInstrumentationLevel", "tile"},
    //     {"debug.outputAllSymbols", "true"},
    //     {"autoReport.all", "true"},
    //     {"autoReport.outputSerializedGraph", "true"},
    //     {"debug.retainDebugInformation", "true"},
    // };

    // auto programIds = map<string, int>();
    // auto programsList = vector<Program>(programs.size());
    // int index = 0;
    // for (auto &nameToProgram : programs)
    // {
    //     programIds[nameToProgram.first] = index;
    //     programsList[index] = nameToProgram.second;
    //     index++;
    // }
    // auto engine = Engine(graph, programsList, ENGINE_OPTIONS);

    // std::cout << "STEP 6: Load compiled graph onto the IPU tiles" << std::endl;
    // engine.load(*device);
    // engine.enableExecutionProfiling();

    // std::cout << "STEP 7: Attach data streams" << std::endl;
    // auto hostData = vector<float>(NUM_DATA_ITEMS, 0.0f);
    // engine.connectStream("TO_IPU", hostData.data());
    // engine.connectStream("FROM_IPU", hostData.data());

    // std::cout << "STEP 8: Run programs" << std::endl;
    // engine.run(programIds["copy_to_ipu"]);  // Copy to IPU
    // engine.run(programIds["main"]);         // Main program
    // engine.run(programIds["copy_to_host"]); // Copy from IPU

    // std::cout << "STEP 9: Capture debug and profile info" << std::endl;
    // serializeGraph(graph);
    // engine.printProfileSummary(std::cout,
    //                            OptionFlags{{"showExecutionSteps", "false"}});

    // return EXIT_SUCCESS;
}