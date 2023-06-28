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
#include <poplar/CycleCount.hpp>
#include <popops/AllTrue.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
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
using ::poplar::SyncType;

using ::poplar::FLOAT;
using ::poplar::INT;
using ::poplar::UNSIGNED_INT;
using ::poplar::BOOL;

using ::poplar::program::Copy;
using ::poplar::program::RepeatWhileFalse;
using ::poplar::program::Execute;
using ::poplar::program::Program;
using ::poplar::program::Repeat;
using ::poplar::program::Sequence;

using ::popops::SingleReduceOp;
using ::popops::reduceMany;

namespace exp_bfs
{

    // Helper functions for experiment
    namespace
    {
        struct IPUMatrix
        {
        public:
            IPUMatrix(vector<int> offsets, vector<int> idx, vector<int> row_idx, int blocks, int block_height, int m, int n, unsigned int frontier) : offsets(offsets), idx(idx), row_idx(row_idx), blocks(blocks), block_height(block_height), m(m), n(n), frontier(frontier) {}

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

        auto prepare_data(matrix::SparseMatrix<float> matrix, const int num_tiles)
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

            vector<int> idx(matrix.nonzeroes());

            for (auto o = 0; o < matrix.nonzeroes(); o++)
            {
                auto [i, j, v] = matrix.get(o);
                (void)v;

                auto x = j / block_size_col;
                auto y = i / block_size_row;

                size_t value_offset = offsets[y * blocks + x] + cursor[(y * blocks + x) * (block_size_row + 1) + (i - (block_size_row * y))];

                idx[value_offset] = j - (block_size_col * x);

                // Update cursor
                cursor[(y * blocks + x) * (block_size_row + 1) + (i - (block_size_row * y))]++;
            }

            unsigned int frontier = 0;
            if (matrix.applied_perm.has_value()) {
                frontier = matrix.applied_perm.value().apply(0);
            }

            return IPUMatrix(offsets, idx, row_idx, blocks, block_size_row, matrix.rows(), matrix.cols(), frontier);
        }

        void build_compute_graph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, const int num_tiles, IPUMatrix &ipu_matrix)
        {

            // Static Matrix data
            tensors["idx"] = graph.addVariable(INT, {ipu_matrix.idx.size()}, "idx");
            tensors["row_idx"] = graph.addVariable(INT, {ipu_matrix.blocks, ipu_matrix.blocks, ipu_matrix.block_height + 1}, "row_idx");

            // Input/Output vector
            tensors["vector"] = graph.addVariable(FLOAT, {(unsigned int)ipu_matrix.n}, "vector");
            graph.setInitialValue(tensors["vector"].slice(0, ipu_matrix.n), poplar::ArrayRef(vector<float>(ipu_matrix.n, 0.0)));
            graph.setInitialValue(tensors["vector"][ipu_matrix.frontier], 1.0); // our first frontier value is always node 0 (col 0)

            tensors["res"] = graph.addVariable(FLOAT, {ipu_matrix.blocks, ipu_matrix.blocks, ipu_matrix.block_height}, "result");

            // We build the compute set for the MatrixBlock codelet
            auto spmv_cs = graph.addComputeSet("spmv");

            for (unsigned int y = 0; y < ipu_matrix.blocks; y++)
            {
                for (unsigned int x = 0; x < ipu_matrix.blocks; x++)
                {
                    auto block_id = y * ipu_matrix.blocks + x;
                    auto v = graph.addVertex(spmv_cs, "MatrixBlock", {
                        {"idx", tensors["idx"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1])},
                        {"row_idx", tensors["row_idx"][y][x]},
                        {"vec", tensors["vector"].slice(std::min(ipu_matrix.m, x * ipu_matrix.block_height), std::min(ipu_matrix.m, (x + 1) * ipu_matrix.block_height))},
                        {"res", tensors["res"][y][x]}
                    });

                    // TODO need to be calculated;
                    graph.setPerfEstimate(v, 100);
                    graph.setTileMapping(v, block_id);

                    graph.setTileMapping(tensors["idx"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1]), block_id);
                    graph.setTileMapping(tensors["row_idx"][y][x], block_id);
                    graph.setTileMapping(tensors["vector"].slice(std::min(ipu_matrix.m, x * ipu_matrix.block_height), std::min(ipu_matrix.m, (x + 1) * ipu_matrix.block_height)), block_id);
                    graph.setTileMapping(tensors["res"][y][x], block_id);
                }
            }

            auto program_spmv = Execute(spmv_cs);

            // We build the compute set for addition
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

                    out.push_back(tensors["vector"][block * ipu_matrix.block_height + y]);
                }
            }

            auto program_reduce = Sequence{};
            popops::reduceMany(graph, reductions, out, program_reduce);

            // So far this is a basic copy of SpMV, now the magic:
            // We need to:
            // 1. Keep track of a dist, and iteration tensor
            // 2. Copy over `vector` (if v[i] > 1 ==> dist[i] = min(dist[i], iteration)) to dist
            // 3. Normalize vector again (ones)
            // 4. Keep track if we should continue! (We perform some sub-reductions to speed up this process)

            tensors["dist"] = graph.addVariable(UNSIGNED_INT, {(unsigned int)ipu_matrix.n}, "dist");
            graph.setInitialValue(tensors["dist"], poplar::ArrayRef(vector<unsigned int>(ipu_matrix.n, UINT_MAX)));
            graph.setInitialValue(tensors["dist"][ipu_matrix.frontier], 0);

            tensors["iteration"] = graph.addVariable(UNSIGNED_INT, {1}, "iteration");
            graph.setTileMapping(tensors["iteration"], 0);
            graph.setInitialValue(tensors["iteration"][0], 1);

            tensors["stop"] = graph.addVariable(BOOL, {(unsigned long)num_tiles}, "stop condition");
            //graph.setInitialValue(tensors["stop"], poplar::ArrayRef(vector<char>(num_tiles, false))); // We use char here because a bool is a 1-bit value in cpp

            auto normalize_cs = graph.addComputeSet("normalize");

            unsigned int rows_per_tile = std::max(ipu_matrix.m / num_tiles + (ipu_matrix.m % num_tiles != 0), (unsigned int) 1);
            for (unsigned int i = 0; i < static_cast<unsigned int>(num_tiles); i++) {
                unsigned int row_start = std::min(i * rows_per_tile, ipu_matrix.m);
                unsigned int row_end = std::min((i + 1) * rows_per_tile, ipu_matrix.m);

                auto v = graph.addVertex(normalize_cs, "Normalize", {
                    {"vec", tensors["vector"].slice(row_start, row_end)},
                    {"dist", tensors["dist"].slice(row_start, row_end)},
                    {"iteration", tensors["iteration"][0]},
                    {"stop", tensors["stop"][i]}
                });

                graph.setPerfEstimate(v, 100);
                graph.setTileMapping(v, i);

                graph.setTileMapping(tensors["stop"][i], i);
                graph.setTileMapping(tensors["dist"].slice(row_start, row_end), i);
            }

            auto program_normalize = Execute(normalize_cs);

            // Subreduce stop tensor
            // unsigned int subreduction_size = (unsigned int)ceil(sqrt((float)num_tiles));
            // tensors["sub_stop"] = graph.addVariable(BOOL, {subreduction_size}, "sub stop condition");
            // poputil::mapTensorLinearly(graph, tensors["sub_stop"]);

            // vector<SingleReduceOp> stop_reductions;
            // stop_reductions.reserve(subreduction_size);
            // vector<Tensor> stop_out;
            // stop_out.reserve(subreduction_size);

            // for (unsigned int i = 0; i < subreduction_size; i++)
            // {
            //     stop_reductions.push_back(SingleReduceOp {
            //         tensors["stop"].slice(std::min(subreduction_size * i, (unsigned int)num_tiles), std::min(subreduction_size * (i + 1), (unsigned int)num_tiles)), {0}, {popops::Operation::LOGICAL_AND}
            //     });

            //     stop_out.push_back(tensors["sub_stop"][i]);
            // }

            // auto program_stop_reduce = Sequence("pre-reduce stop variable");
            // popops::reduceMany(graph, stop_reductions, stop_out, program_stop_reduce);

            auto program_sequence = Sequence{
                program_spmv, program_reduce, program_normalize
            };

            tensors["should_stop"] = popops::allTrue(graph, tensors["stop"], program_sequence, "check stop condition");
            popops::mapInPlace(graph, popops::expr::_1 + popops::expr::Const(1), {tensors["iteration"][0]}, program_sequence, "add 1 to iteration");

            auto main_sequence = Sequence{RepeatWhileFalse(Sequence(), tensors["should_stop"], program_sequence)};

            if (!Config::get().model)
            {
                auto timing = poplar::cycleCount(graph, main_sequence, 0, SyncType::INTERNAL, "timer");
                graph.createHostRead("readTimer", timing, true);
            }

            programs["main"] = main_sequence;
        }

        auto build_data_streams(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, IPUMatrix &ipu_matrix)
        {
            auto toipu_idx = graph.addHostToDeviceFIFO("toipu_idx", INT, ipu_matrix.idx.size());
            auto toipu_row_idx = graph.addHostToDeviceFIFO("toipu_row_idx", INT, ipu_matrix.row_idx.size());

            auto fromipu_dist = graph.addDeviceToHostFIFO("fromipu_dist", UNSIGNED_INT, ipu_matrix.n);

            auto copyto_idx = Copy(toipu_idx, tensors["idx"]);
            auto copyto_row_idx = Copy(toipu_row_idx, tensors["row_idx"]);

            auto copyhost_vec = Copy(tensors["dist"], fromipu_dist);

            programs["copy_to_ipu_matrix"] = Sequence{copyto_idx, copyto_row_idx};
            programs["copy_to_host"] = copyhost_vec;
        }

        auto create_graph_add_codelets(const Device &device) -> Graph
        {
            auto graph = poplar::Graph(device.getTarget());

            // Add our own codelets
            graph.addCodelets({"codelets/bfs/MatrixBlock.cpp", "codelets/bfs/Normalize.cpp"}, "-O3 -I codelets");
            popops::addCodelets(graph);

            return graph;
        }
    }

    optional<ExperimentReportIPU> execute(const Device &device, matrix::SparseMatrix<float> &matrix)
    {
        std::cerr << "Executing BFS experiment.." << std::endl;

        Graph graph = create_graph_add_codelets(device);

        auto tensors = map<string, Tensor>{};
        auto programs = map<string, Program>{};

        auto ipu_matrix = prepare_data(matrix, device.getTarget().getNumTiles());

        std::cerr << "Building programs.." << std::endl;

        build_compute_graph(graph, tensors, programs, device.getTarget().getNumTiles(), ipu_matrix);
        build_data_streams(graph, tensors, programs, ipu_matrix);

        auto ENGINE_OPTIONS = OptionFlags{};

        if (Config::get().debug)
        {
            ENGINE_OPTIONS = OptionFlags{
                {"autoReport.all", "true"}};
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

        std::cerr << "Compiling graph.." << std::endl;
        
        auto timing_graph_compilation_start = std::chrono::high_resolution_clock::now();
        auto engine = Engine(graph, programsList, ENGINE_OPTIONS);
        engine.load(device);
        auto timing_graph_compilation_end = std::chrono::high_resolution_clock::now();
        auto timing_graph_compilation = std::chrono::duration_cast<std::chrono::nanoseconds>(timing_graph_compilation_end - timing_graph_compilation_start).count() / 1e3;

        if (Config::get().debug)
        {
            engine.enableExecutionProfiling();
        }

        auto vec = vector<float>(ipu_matrix.n, 1.0);

        // TODO: if we change the input vector we need to apply the matrix mapping to it for a correct result.

        engine.connectStream("toipu_idx", ipu_matrix.idx.data());
        engine.connectStream("toipu_row_idx", ipu_matrix.row_idx.data());

        auto result_dist = vector<unsigned int>(ipu_matrix.n);
        engine.connectStream("fromipu_dist", result_dist.data());

        // Run all programs in order
        std::cerr << "Running programs.." << std::endl;
        std::cerr << "Copy data to IPU\n";

        auto copy_timing_start = std::chrono::high_resolution_clock::now();
        engine.run(programIds["copy_to_ipu_matrix"], "copy matrix");
        auto copy_timing_end = std::chrono::high_resolution_clock::now();
        auto copy_timing = std::chrono::duration_cast<std::chrono::nanoseconds>(copy_timing_end - copy_timing_start).count() / 1e3;

        std::cerr << "Run main program\n";

        auto execution_start = std::chrono::high_resolution_clock::now();
        engine.run(programIds["main"], "main loop");
        auto execution_end = std::chrono::high_resolution_clock::now();
        auto execution_timing = std::chrono::duration_cast<std::chrono::nanoseconds>(execution_end - execution_start).count() / 1e3;

        vector<unsigned long> ipuTimer(1);
        if (!Config::get().model)
        {
            engine.readTensor("readTimer", ipuTimer.data(), &*ipuTimer.end());
            std::cerr << "Timing read: " << ipuTimer[0] << std::endl;
        }

        std::cerr << "Copying back result\n"; 

        auto copyback_timing_start = std::chrono::high_resolution_clock::now();
        engine.run(programIds["copy_to_host"], "copy result");
        auto copyback_timing_end = std::chrono::high_resolution_clock::now();
        auto copyback_timing = std::chrono::duration_cast<std::chrono::nanoseconds>(copyback_timing_end - copyback_timing_start).count() / 1e3;

        std::cerr << "Resulting vector:\n";
        long int res = 0;
        for (auto v : result_dist)
        {
            std::cerr << v << ", ";
            res += static_cast<long int>(v);
        }
        std::cerr << std::endl;
        std::cerr << "Sum: " << res << std::endl;

        // setup result report
        auto report = ExperimentReportIPU(std::move(engine), std::move(graph));
        report.set_timing("copy", copy_timing);
        report.set_timing("execution", execution_timing);
        report.set_timing("copy_back", copyback_timing);
        report.set_timing("graph_compilation", timing_graph_compilation);

        if (!Config::get().model)
        {
            report.set_timing("ipu_report", ipuTimer[0] / device.getTarget().getTileClockFrequency());
        }

        return optional(std::move(report));
    }
}