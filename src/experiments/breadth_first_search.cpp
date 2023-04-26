#include <cstdlib>
#include <algorithm>
#include <cmath>

#include <poplar/Engine.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <popops/Loop.hpp>

#include "../matrix.hpp"
#include "../config.cpp"
#include "../ipu.cpp"

using ::poplar::Device;
using ::poplar::Engine;
using ::poplar::Graph;
using ::poplar::Tensor;
using ::poplar::OptionFlags;

using ::poplar::FLOAT;
using ::poplar::INT;
using ::poplar::UNSIGNED_INT;
using ::poplar::BOOL;

using ::poplar::program::Copy;
using ::poplar::program::Execute;
using ::poplar::program::Program;
using ::poplar::program::Repeat;
using ::poplar::program::Sequence;
using ::poplar::program::PrintTensor;

namespace exp_bfs
{
    const int MAX_ITERATIONS = 5000;

    // Helper functions for experiment
    namespace
    {
        struct BFS_IPUMatrix
        {
        public:
            BFS_IPUMatrix(vector<int> offsets, vector<int> idx, vector<int> row_idx, int blocks, int block_height, int m, int n) : offsets(offsets), idx(idx), row_idx(row_idx), blocks(blocks), block_height(block_height), m(m), n(n) {}

            vector<int> offsets;

            // actuall data
            vector<int> idx; // better indexing type? size_t?
            vector<int> row_idx;

            // matrix data
            unsigned int blocks;
            unsigned int block_height;
            unsigned int m;
            unsigned int n;
        };

        auto prepare_data(matrix::Matrix<float> matrix, const int num_tiles)
        {
            // assumptions at this point in the code: matrix is shuffled (values are normally divided)
            // TODO: prepareData currently takes O(n*m), can be done in O(nz) for SparseMatrix type

            // First we calculate how many blocks we have available. We need x tiles for summation and x^2 blocks for the SpMV
            // In general this _should_ make it possible to execute SpMV on the same matrix twice with differents vectors.
            const auto blocks = (int)std::floor((-1.0 + std::sqrt(1 + 4 * num_tiles)) / 2.0); // For a standard IPU 37*37
            const auto block_size_col = std::max(matrix.cols() / blocks + (matrix.cols() % blocks != 0), 1);
            const auto block_size_row = std::max(matrix.rows() / blocks + (matrix.rows() % blocks != 0), 1);

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
                                idx[offsets[y * blocks + x + 1]] = mj - block_size_col * x;
                                offsets[y * blocks + x + 1] += 1;
                            }
                        }
                    }

                    row_idx[(y * blocks + x) * (block_size_row + 1) + std::min(block_size_row * (y + 1), matrix.rows()) - block_size_row * y] = offsets[y * blocks + x + 1] - offsets[y * blocks + x];
                }
            }

            // Final value should be nz (sum of every block)
            assert(offsets[offsets.size() - 1] == matrix.nonzeroes());

            return BFS_IPUMatrix(offsets, idx, row_idx, blocks, block_size_row, matrix.rows(), matrix.cols());
        }

        void build_compute_graph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, const int num_tiles, BFS_IPUMatrix &ipu_matrix, const int loops)
        {
            // Static Matrix data
            tensors["idx"] = graph.addVariable(INT, {ipu_matrix.idx.size()}, "idx");
            poputil::mapTensorLinearly(graph, tensors["idx"]);
            tensors["row_idx"] = graph.addVariable(INT, {ipu_matrix.row_idx.size()}, "row_idx");
            poputil::mapTensorLinearly(graph, tensors["row_idx"]);

            // Input/Output vector
            tensors["frontier"] = graph.addVariable(BOOL, {(unsigned int)ipu_matrix.n}, "frontier");
            poputil::mapTensorLinearly(graph, tensors["frontier"]);

            tensors["dist"] = graph.addVariable(UNSIGNED_INT, {(unsigned int)ipu_matrix.n}, "dist");
            poputil::mapTensorLinearly(graph, tensors["dist"]);

            tensors["res"] = graph.addVariable(BOOL, {(unsigned int)ipu_matrix.blocks * ipu_matrix.blocks * ipu_matrix.block_height}, "result");
            poputil::mapTensorLinearly(graph, tensors["res"]);

            tensors["iteration"] = graph.addVariable(UNSIGNED_INT, {1}, "iteration");
            graph.setTileMapping(tensors["iteration"], ipu_matrix.blocks * ipu_matrix.blocks);

            tensors["max_iteration"] = graph.addConstant<int>(UNSIGNED_INT, {1}, {MAX_ITERATIONS}, "max_iteration");
            graph.setTileMapping(tensors["max_iteration"], ipu_matrix.blocks * ipu_matrix.blocks);

            // We build the compute set for the MatrixBlock codelet
            auto spmv_cs = graph.addComputeSet("spmv");

            for (unsigned int y = 0; y < ipu_matrix.blocks; y++)
            {
                for (unsigned int x = 0; x < ipu_matrix.blocks; x++)
                {
                    auto block_id = y * ipu_matrix.blocks + x;
                    auto v = graph.addVertex(spmv_cs, "MatrixBlockBFS", {
                        {"idx", tensors["idx"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1])},
                        {"row_idx", tensors["row_idx"].slice((y * ipu_matrix.blocks + x) * (ipu_matrix.block_height + 1), (y * ipu_matrix.blocks + x + 1) * (ipu_matrix.block_height + 1))},
                        {"frontier", tensors["frontier"]},
                        {"res", tensors["res"].slice(block_id * ipu_matrix.block_height, (block_id + 1) * ipu_matrix.block_height)}
                    });

                    // TODO need to be calculated;
                    graph.setPerfEstimate(v, 100);
                    graph.setTileMapping(v, block_id);

                    // graph.setTileMapping(tensors["idx"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1]), block_id);
                    // graph.setTileMapping(tensors["row_idx"].slice((y * ipu_matrix.blocks + x) * (ipu_matrix.block_height + 1), (y * ipu_matrix.blocks + x + 1) * (ipu_matrix.block_height + 1)), block_id);
                    // graph.setTileMapping(tensors["res"].slice(block_id * ipu_matrix.block_height, (block_id + 1) * ipu_matrix.block_height), block_id);
                }
            }

            // We build the compute set for addition
            auto reducer_cs = graph.addComputeSet("reduce");

            for (unsigned int y = 0; y < ipu_matrix.blocks; y++) {
                
                auto v = graph.addVertex(reducer_cs, "ReducerBFS", {
                    {"res", tensors["res"].slice(y * ipu_matrix.blocks * ipu_matrix.block_height, (y + 1) * ipu_matrix.blocks * ipu_matrix.block_height)},
                    {"next_frontier", tensors["frontier"].slice(std::min(ipu_matrix.m, y * ipu_matrix.block_height), std::min(ipu_matrix.m, (y + 1) * ipu_matrix.block_height))},
                    {"dist", tensors["dist"].slice(std::min(ipu_matrix.m, y * ipu_matrix.block_height), std::min(ipu_matrix.m, (y + 1) * ipu_matrix.block_height))},
                    {"iteration", tensors["iteration"][0]}
                });

                graph.setInitialValue(v["block_length"], std::min((int)ipu_matrix.block_height, std::max(0, (int)ipu_matrix.m - (int)ipu_matrix.block_height * (int)y)));
                graph.setInitialValue(v["res_block_length"], ipu_matrix.block_height); // TODO not necessary if we compute res better
                graph.setInitialValue(v["blocks"], ipu_matrix.blocks);

                graph.setPerfEstimate(v, 100);
                graph.setTileMapping(v, ipu_matrix.blocks * ipu_matrix.blocks + y);
                // graph.setTileMapping(tensors["dist"].slice(std::min(ipu_matrix.m, y * ipu_matrix.block_height), std::min(ipu_matrix.m, (y + 1) * ipu_matrix.block_height)), ipu_matrix.blocks * ipu_matrix.blocks + y);
            }

            programs["main"] = popops::countedForLoop(graph, tensors["iteration"], 1, tensors["max_iteration"], 1, Sequence{Execute(spmv_cs), Execute(reducer_cs)});
        }

        auto build_data_streams(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, BFS_IPUMatrix &ipu_matrix)
        {
            auto toipu_idx = graph.addHostToDeviceFIFO("toipu_idx", INT, ipu_matrix.idx.size());
            auto toipu_row_idx = graph.addHostToDeviceFIFO("toipu_row_idx", INT, ipu_matrix.row_idx.size());
            auto toipu_frontier = graph.addHostToDeviceFIFO("toipu_frontier", BOOL, ipu_matrix.n);
            auto toipu_dist = graph.addHostToDeviceFIFO("toipu_dist", UNSIGNED_INT, ipu_matrix.n);

            auto fromipu_dist = graph.addDeviceToHostFIFO("fromipu_dist", UNSIGNED_INT, ipu_matrix.n);

            auto copyto_idx = Copy(toipu_idx, tensors["idx"]);
            auto copyto_row_idx = Copy(toipu_row_idx, tensors["row_idx"]);
            auto copyto_frontier = Copy(toipu_frontier, tensors["frontier"]);
            auto copyto_dist = Copy(toipu_dist, tensors["dist"]);

            auto copyhost_dist = Copy(tensors["dist"], fromipu_dist);

            programs["copy_to_ipu_matrix"] = Sequence{copyto_idx, copyto_row_idx};
            programs["copy_to_ipu_frontier"] = Sequence{copyto_dist, copyto_frontier};

            programs["copy_to_host"] = copyhost_dist;
        }

        auto create_graph_add_codelets(const Device &device) -> Graph
        {
            auto graph = poplar::Graph(device.getTarget());

            // Add our own codelets
            graph.addCodelets({"codelets/bfs/MatrixBlockBFS.cpp", "codelets/bfs/ReducerBFS.cpp"}, "-O3 -I codelets");
            popops::addCodelets(graph);

            return graph;
        }
    }

    // struct ExpResult {
    //     Graph graph;
    //     Engine &engine;

    //     ExpResult(Graph g, Engine &e): graph(g), engine(e) {};
    // };

    auto execute(const Device &device, matrix::Matrix<float> &matrix, int rounds)
    {
        std::cout << "Executing BFS experiment.." << std::endl;

        Graph graph = create_graph_add_codelets(device);

        auto tensors = map<string, Tensor>{};
        auto programs = map<string, Program>{};

        auto ipu_matrix = prepare_data(matrix, device.getTarget().getNumTiles());

        std::cout << "Building programs.." << std::endl;

        build_compute_graph(graph, tensors, programs, device.getTarget().getNumTiles(), ipu_matrix, rounds);
        build_data_streams(graph, tensors, programs, ipu_matrix);

        auto ENGINE_OPTIONS = OptionFlags{};

        if (Config::get().debug)
        {
            ENGINE_OPTIONS = OptionFlags{
                {"autoReport.all", "true"}
            };
        }

        auto programIds = map<string, int>();
        auto programsList = vector<Program>(programs.size());
        int index = 0;
        for (auto &nameToProgram : programs)
        {
            programIds[nameToProgram.first] = index;
            programsList[index] = nameToProgram.second;
            index++;
        }

        std::cout << "Compiling graph.." << std::endl;
        auto engine = Engine(graph, programsList, ENGINE_OPTIONS);
        engine.load(device);

        if (Config::get().debug)
        {
            engine.enableExecutionProfiling();
        }

        auto frontier = vector<int>(ipu_matrix.n, false);
        frontier[0] = 1;

        engine.connectStream("toipu_idx", ipu_matrix.idx.data());
        engine.connectStream("toipu_row_idx", ipu_matrix.row_idx.data());
        engine.connectStream("toipu_frontier", frontier.data());
        engine.connectStream("toipu_dist", frontier.data());

        auto result_vec = vector<int>(ipu_matrix.n);
        engine.connectStream("fromipu_dist", result_vec.data());

        // Copy data
        std::cout << "Running programs.." << std::endl;
        engine.run(programIds["copy_to_ipu_matrix"], "copy matrix");
        engine.run(programIds["copy_to_ipu_vec"], "copy vec");

        std::cout << "Running main loop.." << std::endl;
        engine.run(programIds["main"], "bfs-spmv");

        // Copy result
        engine.run(programIds["copy_to_host"], "copy result");

        std::cout << "Resulting vector:\n";
        for (auto v : result_vec)
        {
            std::cout << v << ", ";
        }
        std::cout << std::endl;

        serialize_graph(graph);
        engine.printProfileSummary(std::cout, OptionFlags{});

        return graph;
    }
}