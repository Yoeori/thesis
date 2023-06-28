#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits.h>

#include <poplar/Engine.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <poplar/ArrayRef.hpp>
#include <popops/ElementWise.hpp>
#include <popops/AllTrue.hpp>
#include <popops/codelets.hpp>
#include <popops/Loop.hpp>
#include <popops/Reduce.hpp>

#include "../matrix.hpp"
#include "../config.cpp"
#include "../ipu.cpp"
#include "../report.cpp"

using ::poplar::Device;
using ::poplar::Engine;
using ::poplar::Graph;
using ::poplar::Tensor;
using ::poplar::OptionFlags;

using ::poplar::FLOAT;
using ::poplar::INT;
using ::poplar::UNSIGNED_INT;
using ::poplar::BOOL;
using ::poplar::SHORT;

using ::poplar::program::Copy;
using ::poplar::program::RepeatWhileFalse;
using ::poplar::program::Execute;
using ::poplar::program::Program;
using ::poplar::program::Repeat;
using ::poplar::program::Sequence;
using ::poplar::program::PrintTensor;

using ::popops::SingleReduceOp;
using ::popops::reduceMany;

namespace exp_bfs
{
    // Helper functions for experiment
    namespace
    {
        struct BFS_IPUMatrix
        {
        public:
            BFS_IPUMatrix(vector<int> offsets, vector<int> idx, vector<int> row_idx, int blocks, int block_height, int m, int n, unsigned int frontier) : offsets(offsets), idx(idx), row_idx(row_idx), blocks(blocks), block_height(block_height), m(m), n(n), frontier(frontier) {}

            vector<int> offsets;

            // actuall data
            vector<int> idx; // better indexing type? size_t?
            vector<int> row_idx;

            // matrix data
            unsigned int blocks;
            unsigned int block_height;
            unsigned int m;
            unsigned int n;

            unsigned int frontier;
        };

        template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
        auto prepare_data(matrix::SparseMatrix<T> matrix, const int num_tiles)
        {
            const auto blocks = (int) std::floor(std::sqrt(num_tiles));
            const auto block_size_col = std::max(matrix.cols() / blocks + (matrix.cols() % blocks != 0), 1);
            const auto block_size_row = std::max(matrix.rows() / blocks + (matrix.rows() % blocks != 0), 1);

            // This will give the offsets for matrix and idx
            vector<int> offsets(blocks * blocks + 1, 0);
            vector<int> row_idx(blocks * blocks * (block_size_row + 1), 0);

            // We go through each value in the SpM and update offsets and row_idx
            for (auto o = 0; o < matrix.nonzeroes(); o++)
            {
                auto [i, j, v] = matrix.get(o);
                (void)v;

                auto x = j / block_size_col;
                auto y = i / block_size_row;

                offsets[y * blocks + x + 1]++;
                row_idx[(y * blocks + x) * (block_size_row + 1) + (i - (block_size_row * y)) + 1]++;
            }

            // Stride offsets and row_idx
            for (size_t i = 2; i < offsets.size(); i++)
            {
                offsets[i] += offsets[i - 1];
            }

            for (auto block = 0; block < blocks * blocks; block++)
            {
                for (auto i = 0; i < block_size_row; i++)
                {
                    row_idx[block * (block_size_row + 1) + i + 1] += row_idx[block * (block_size_row + 1) + i];
                }
            }

            assert(offsets[offsets.size() - 1] == matrix.nonzeroes());

            vector<int> cursor(row_idx); // Cursor inside a block between rows

            vector<T> ipu_matrix(matrix.nonzeroes());
            vector<int> idx(matrix.nonzeroes());

            for (auto o = 0; o < matrix.nonzeroes(); o++)
            {
                auto [i, j, v] = matrix.get(o);

                auto x = j / block_size_col;
                auto y = i / block_size_row;

                size_t value_offset = offsets[y * blocks + x] + cursor[(y * blocks + x) * (block_size_row + 1) + (i - (block_size_row * y))];

                ipu_matrix[value_offset] = v;
                idx[value_offset] = j - (block_size_col * x);

                // Update cursor
                cursor[(y * blocks + x) * (block_size_row + 1) + (i - (block_size_row * y))]++;
            }

            unsigned int frontier = 0;

            if (matrix.applied_perm.has_value()) {
                frontier = matrix.applied_perm.value().apply(0);
            }

            return BFS_IPUMatrix(offsets, idx, row_idx, blocks, block_size_row, matrix.rows(), matrix.cols(), frontier);
        }

        void build_compute_graph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, const int num_tiles, BFS_IPUMatrix &ipu_matrix)
        {
            // Static Matrix data
            tensors["idx"] = graph.addVariable(INT, {ipu_matrix.idx.size()}, "idx");
            tensors["row_idx"] = graph.addVariable(INT, {ipu_matrix.blocks, ipu_matrix.blocks, ipu_matrix.block_height + 1}, "row_idx");

            // Input/Output vector
            tensors["frontier"] = graph.addVariable(BOOL, {(unsigned int)ipu_matrix.n}, "frontier");
            graph.setInitialValue(tensors["frontier"].slice(0, ipu_matrix.n), poplar::ArrayRef(vector<char>(ipu_matrix.n, 1)));
            graph.setInitialValue(tensors["frontier"][ipu_matrix.frontier], false); // our first frontier value is always node 0 (col 0)

            tensors["dist"] = graph.addVariable(UNSIGNED_INT, {(unsigned int)ipu_matrix.n}, "dist");
            graph.setInitialValue(tensors["dist"], poplar::ArrayRef(vector<int>(ipu_matrix.n, INT_MAX)));
            graph.setInitialValue(tensors["dist"][ipu_matrix.frontier], 0);
            poputil::mapTensorLinearly(graph, tensors["dist"]);

            // There seem to be memory issues when using booleans for such big data structures. We therefore use unsigned int instead
            // This does result in 4x more memory use
            tensors["res"] = graph.addVariable(UNSIGNED_INT, {ipu_matrix.blocks, ipu_matrix.blocks, ipu_matrix.block_height}, "result");

            // Loop tensors
            tensors["iteration"] = graph.addVariable(UNSIGNED_INT, {1}, "iteration");
            graph.setTileMapping(tensors["iteration"], 0);
            graph.setInitialValue(tensors["iteration"][0], 1);

            tensors["stop"] = graph.addVariable(BOOL, {(unsigned long)num_tiles}, "stop condition");
            graph.setTileMapping(tensors["stop"], 0);
            graph.setInitialValue(tensors["stop"], poplar::ArrayRef(vector<char>(num_tiles, true)));

            // We build the compute set for the MatrixBlock codelet
            auto spmv_cs = graph.addComputeSet("spmv");

            for (unsigned int y = 0; y < ipu_matrix.blocks; y++)
            {
                for (unsigned int x = 0; x < ipu_matrix.blocks; x++)
                {
                    auto block_id = y * ipu_matrix.blocks + x;
                    auto v = graph.addVertex(spmv_cs, "MatrixBlockBFS", {
                        {"idx", tensors["idx"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1])},
                        {"row_idx", tensors["row_idx"][y][x]},
                        {"frontier", tensors["frontier"].slice(std::min(ipu_matrix.m, x * ipu_matrix.block_height), std::min(ipu_matrix.m, (x + 1) * ipu_matrix.block_height))},
                        {"res", tensors["res"][y][x]},
                        {"iteration", tensors["iteration"][0]}
                    });

                    // TODO need to be calculated;
                    graph.setPerfEstimate(v, 100);
                    graph.setTileMapping(v, block_id);

                    graph.setTileMapping(tensors["idx"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1]), block_id);
                    graph.setTileMapping(tensors["row_idx"][y][x], block_id);
                    graph.setTileMapping(tensors["frontier"].slice(std::min(ipu_matrix.m, x * ipu_matrix.block_height), std::min(ipu_matrix.m, (x + 1) * ipu_matrix.block_height)), block_id);
                    graph.setTileMapping(tensors["res"][y][x], block_id);
                }
            }

            // We build the compute set for addition
            // auto reducer_cs = graph.addComputeSet("reduce");
            // auto res_vector_shuffled = tensors["res"].dimShuffle({0, 2, 1}).reshape({ipu_matrix.blocks * ipu_matrix.block_height, ipu_matrix.blocks});
            // auto rows_per_vertex = std::max(ipu_matrix.n / num_tiles + (ipu_matrix.n % num_tiles != 0), (unsigned int)1);

            // for (int i = 0; i < num_tiles; i++)
            // {
            //     auto start = std::min(rows_per_vertex * i, ipu_matrix.m);
            //     auto end = std::min(rows_per_vertex * (i + 1), ipu_matrix.m);

            //     auto v = graph.addVertex(reducer_cs, "ReducerRowBFSMulti", {
            //         {"dist", tensors["dist"].slice(start, end)},
            //         {"frontier", tensors["frontier"].slice(start, end)},
            //         {"res", res_vector_shuffled.slice(start, end)},
            //         {"iteration", tensors["iteration"][0]}
            //     });

            //     graph.setPerfEstimate(v, 100);
            //     graph.setTileMapping(v, i);
            //     graph.setTileMapping(tensors["dist"].slice(start, end), i);
            // }

            // auto reducer_cs = graph.addComputeSet("reduce");
            // auto res_vector_shuffled = tensors["res"].dimShuffle({0, 2, 1});

            // for (unsigned int y = 0; y < ipu_matrix.blocks; y++)
            // {
            //     for (unsigned int row = 0; row < ipu_matrix.block_height && (y * ipu_matrix.block_height) + row < ipu_matrix.m; row++)
            //     {
            //         auto v = graph.addVertex(reducer_cs, "ReducerRowBFS", {
            //             {"dist", tensors["dist"][(ipu_matrix.block_height * y) + row]},
            //             {"frontier", tensors["frontier"][(ipu_matrix.block_height * y) + row]},
            //             {"res", res_vector_shuffled[y][row]},
            //             {"iteration", tensors["iteration"][0]}
            //         });

            //         auto block_id = row % ipu_matrix.blocks + ipu_matrix.blocks * y;

            //         graph.setPerfEstimate(v, 100);
            //         graph.setTileMapping(v, block_id);
            //         graph.setTileMapping(tensors["dist"][(ipu_matrix.block_height * y) + row], block_id);
            //     }
            // }

            // AND reduction over frontier to speedup while check
            auto res_vector_shuffled = tensors["res"].dimShuffle({0, 2, 1});

            vector<SingleReduceOp> reductions;
            reductions.reserve(ipu_matrix.m); // One reduction for every row of our matrix

            vector<Tensor> out;
            out.reserve(ipu_matrix.m);

            for (unsigned int block = 0; block < ipu_matrix.blocks; block++)
            {
                for (unsigned int y = 0; y < ipu_matrix.block_height && block * ipu_matrix.block_height + y < ipu_matrix.m; y++)
                {
                    reductions.push_back(SingleReduceOp {
                        res_vector_shuffled[block][y], {0}, {popops::Operation::ADD}
                    });

                    out.push_back(tensors["dist"][block * ipu_matrix.block_height + y]);
                }
            }

            auto reduce = Sequence{};
            popops::reduceMany(graph, reductions, out, reduce);

            // vector<SingleReduceOp> reductions;
            // reductions.reserve(num_tiles);

            // vector<Tensor> out;
            // out.reserve(num_tiles);

            // size_t per_reduce = std::max(ipu_matrix.n / num_tiles + (ipu_matrix.n % num_tiles != 0), (unsigned int)1);

            // for (auto i = 0; i < num_tiles; i++)
            // {
            //     reductions.push_back(SingleReduceOp {
            //         tensors["frontier"].slice(std::min(per_reduce * i, (size_t)ipu_matrix.n), std::min(per_reduce * (i + 1), (size_t)ipu_matrix.n)), {0}, {popops::Operation::LOGICAL_AND}
            //     });

            //     out.push_back(tensors["stop"][i]);
            //     graph.setTileMapping(tensors["stop"][i], i);
            // }

            // auto p = Sequence{};
            // popops::reduceMany(graph, reductions, out, p, "subreduce stop");

            // for (unsigned int y = 0; y < ipu_matrix.blocks; y++) {

            //     auto block = y * ipu_matrix.blocks;
                
            //     auto v = graph.addVertex(reducer_cs, "ReducerBFS", {
            //         {"res", tensors["res"][y]},
            //         {"frontier", tensors["frontier"].slice(std::min(ipu_matrix.m, y * ipu_matrix.block_height), std::min(ipu_matrix.m, (y + 1) * ipu_matrix.block_height))},
            //         {"dist", tensors["dist"].slice(std::min(ipu_matrix.m, y * ipu_matrix.block_height), std::min(ipu_matrix.m, (y + 1) * ipu_matrix.block_height))},
            //         {"iteration", tensors["iteration"][0]},
            //         {"stop", tensors["stop"][y]}
            //     });

            //     graph.setInitialValue(v["block_length"], std::min((int)ipu_matrix.block_height, std::max(0, (int)ipu_matrix.m - (int)ipu_matrix.block_height * (int)y)));
            //     graph.setInitialValue(v["blocks"], ipu_matrix.blocks);

            //     graph.setPerfEstimate(v, 100);
            //     graph.setTileMapping(v, block);
            //     graph.setTileMapping(tensors["dist"].slice(std::min(ipu_matrix.m, y * ipu_matrix.block_height), std::min(ipu_matrix.m, (y + 1) * ipu_matrix.block_height)), block);
            //     graph.setTileMapping(tensors["stop"][y], block);
            // }

            // Setup program
            auto program_sequence = Sequence{};

            // popops::mapInPlace(graph, popops::expr::_1 + popops::expr::Const(1), {tensors["iteration"]}, program_sequence, "add 1 to iteration");
            program_sequence.add(Execute(spmv_cs));
            program_sequence.add(reduce);
            // program_sequence.add(p);

            // tensors["should_stop"] = popops::allTrue(graph, tensors["stop"][0], program_sequence, "check stop condition");

            // programs["main"] = RepeatWhileFalse(Sequence(), tensors["should_stop"], program_sequence);
            programs["main"] = Repeat(5, program_sequence);
        }

        auto build_data_streams(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, BFS_IPUMatrix &ipu_matrix)
        {
            auto toipu_idx = graph.addHostToDeviceFIFO("toipu_idx", INT, ipu_matrix.idx.size());
            auto toipu_row_idx = graph.addHostToDeviceFIFO("toipu_row_idx", INT, ipu_matrix.row_idx.size());

            auto fromipu_dist = graph.addDeviceToHostFIFO("fromipu_dist", UNSIGNED_INT, ipu_matrix.n);

            auto copyto_idx = Copy(toipu_idx, tensors["idx"]);
            auto copyto_row_idx = Copy(toipu_row_idx, tensors["row_idx"]);
            auto copyhost_dist = Copy(tensors["dist"], fromipu_dist);

            programs["copy_to_ipu_matrix"] = Sequence{copyto_idx, copyto_row_idx};
            programs["copy_to_host"] = copyhost_dist;
        }

        auto create_graph_add_codelets(const Device &device) -> Graph
        {
            auto graph = poplar::Graph(device.getTarget());

            // Add our own codelets
            graph.addCodelets({"codelets/bfs/MatrixBlockBFS.cpp", "codelets/bfs/ReducerRowBFS.cpp", "codelets/bfs/ReducerRowBFSMulti.cpp"}, "-O3 -I codelets");
            popops::addCodelets(graph);

            return graph;
        }
    }

    optional<ExperimentReportIPU> execute(const Device &device, matrix::SparseMatrix<float> &matrix)
    {
        std::cout << "Executing BFS experiment.." << std::endl;

        Graph graph = create_graph_add_codelets(device);

        auto tensors = map<string, Tensor>{};
        auto programs = map<string, Program>{};

        auto ipu_matrix = prepare_data(matrix, device.getTarget().getNumTiles());

        std::cout << "Building programs.." << std::endl;

        build_compute_graph(graph, tensors, programs, device.getTarget().getNumTiles(), ipu_matrix);
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

        engine.connectStream("toipu_idx", ipu_matrix.idx.data());
        engine.connectStream("toipu_row_idx", ipu_matrix.row_idx.data());

        auto result_vec = vector<int>(ipu_matrix.n);
        engine.connectStream("fromipu_dist", result_vec.data());

        // Copy data
        std::cout << "Running programs.." << std::endl;
        engine.run(programIds["copy_to_ipu_matrix"], "copy matrix");

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

        std::cout << "Matrix size: " << matrix.rows() << std::endl;

        auto report = ExperimentReportIPU(std::move(engine), std::move(graph));
        return optional(std::move(report));
    }
}