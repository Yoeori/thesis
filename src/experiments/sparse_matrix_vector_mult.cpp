#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <chrono>

#include <poplar/Engine.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <poplar/CycleCount.hpp>
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

using ::poplar::program::Copy;
using ::poplar::program::Execute;
using ::poplar::program::Program;
using ::poplar::program::Repeat;
using ::poplar::program::Sequence;

using ::popops::SingleReduceOp;
using ::popops::reduceMany;

namespace exp_spmv
{

    // Helper functions for experiment
    namespace
    {
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
            unsigned int blocks;
            unsigned int block_height;
            unsigned int m;
            unsigned int n;
        };

        template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
        auto prepare_data(matrix::Matrix<T> matrix, const int num_tiles)
        {
            // assumptions at this point in the code: matrix is shuffled (values are normally divided)
            // TODO: prepareData currently takes O(n*m), can be done in O(nz) for SparseMatrix type

            // First we calculate how many blocks we have available. We need x tiles for summation and x^2 blocks for the SpMV
            // In general this _should_ make it possible to execute SpMV on the same matrix twice with differents vectors.
            // const auto blocks = (int)std::floor((-1.0 + std::sqrt(1 + 4 * num_tiles)) / 2.0); // For a standard IPU 37*37
            const auto blocks = (int) std::floor(std::sqrt(num_tiles));
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

            return IPUMatrix(offsets, ipu_matrix, idx, row_idx, blocks, block_size_row, matrix.rows(), matrix.cols());
        }

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

            return IPUMatrix(offsets, ipu_matrix, idx, row_idx, blocks, block_size_row, matrix.rows(), matrix.cols());
        }

        void build_compute_graph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, const int num_tiles, IPUMatrix<float> &ipu_matrix, const int loops)
        {

            // Static Matrix data
            tensors["matrix"] = graph.addVariable(FLOAT, {ipu_matrix.matrix.size()}, "matrix");
            tensors["idx"] = graph.addVariable(INT, {ipu_matrix.idx.size()}, "idx");
            tensors["row_idx"] = graph.addVariable(INT, {ipu_matrix.blocks, ipu_matrix.blocks, ipu_matrix.block_height + 1}, "row_idx");

            // Input/Output vector
            tensors["vector"] = graph.addVariable(FLOAT, {(unsigned int)ipu_matrix.n}, "vector");
            tensors["res"] = graph.addVariable(FLOAT, {ipu_matrix.blocks, ipu_matrix.blocks, ipu_matrix.block_height}, "result");

            // We build the compute set for the MatrixBlock codelet
            auto spmv_cs = graph.addComputeSet("spmv");

            for (unsigned int y = 0; y < ipu_matrix.blocks; y++)
            {
                for (unsigned int x = 0; x < ipu_matrix.blocks; x++)
                {
                    auto block_id = y * ipu_matrix.blocks + x;
                    auto v = graph.addVertex(spmv_cs, "MatrixBlock", {
                        {"matrix", tensors["matrix"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1])},
                        {"idx", tensors["idx"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1])},
                        {"row_idx", tensors["row_idx"][y][x]},
                        {"vec", tensors["vector"].slice(std::min(ipu_matrix.m, x * ipu_matrix.block_height), std::min(ipu_matrix.m, (x + 1) * ipu_matrix.block_height))},
                        {"res", tensors["res"][y][x]}
                    });

                    // TODO need to be calculated;
                    graph.setPerfEstimate(v, 100);
                    graph.setTileMapping(v, block_id);

                    graph.setTileMapping(tensors["matrix"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1]), block_id);
                    graph.setTileMapping(tensors["idx"].slice(ipu_matrix.offsets[block_id], ipu_matrix.offsets[block_id + 1]), block_id);
                    graph.setTileMapping(tensors["row_idx"][y][x], block_id);
                    graph.setTileMapping(tensors["vector"].slice(std::min(ipu_matrix.m, x * ipu_matrix.block_height), std::min(ipu_matrix.m, (x + 1) * ipu_matrix.block_height)), block_id);
                    graph.setTileMapping(tensors["res"][y][x], block_id);
                }
            }

            auto program_spmv = Execute(spmv_cs);

            // We build the compute set for addition
            auto reducer_cs = graph.addComputeSet("reduce");
            poplar::program::Program program_reduce;

            if (!Config::get().own_reducer) {

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

                auto p = Sequence{};
                popops::reduceMany(graph, reductions, out, p);
                program_reduce = p;

            } else {
                std::cerr << "Using own reducer, this will lead to a slower execution." << std::endl;

                for (unsigned int y = 0; y < ipu_matrix.blocks; y++)
                {
                    auto v = graph.addVertex(reducer_cs, "ReducerToVector", {
                        {"res", tensors["res"][y]},
                        {"vector", tensors["vector"].slice(std::min(ipu_matrix.m, y * ipu_matrix.block_height), std::min(ipu_matrix.m, (y + 1) * ipu_matrix.block_height))}
                    });

                    graph.setInitialValue(v["block_length"], std::min((int)ipu_matrix.block_height, std::max(0, (int)ipu_matrix.m - (int)ipu_matrix.block_height * (int)y)));
                    graph.setInitialValue(v["blocks"], ipu_matrix.blocks);

                    graph.setPerfEstimate(v, 100);
                    graph.setTileMapping(v, ipu_matrix.blocks * y);
                }

                program_reduce = Execute(reducer_cs);
            }

            auto main_sequence = Sequence{Repeat(
                loops,
                Sequence{program_spmv, program_reduce})};

            if (!Config::get().model)
            {
                auto timing = poplar::cycleCount(graph, main_sequence, 0, SyncType::INTERNAL, "timer");
                graph.createHostRead("readTimer", timing, true);
            }

            programs["main"] = main_sequence;
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

        auto create_graph_add_codelets(const Device &device) -> Graph
        {
            auto graph = poplar::Graph(device.getTarget());

            // Add our own codelets
            graph.addCodelets({"codelets/spmv/MatrixBlock.cpp", "codelets/spmv/ReducerToVector.cpp"}, "-O3 -I codelets");
            popops::addCodelets(graph);

            return graph;
        }
    }

    optional<ExperimentReportIPU> execute(const Device &device, matrix::SparseMatrix<float> &matrix, int rounds)
    {
        std::cerr << "Executing Sparse Matrix Vector multiplication experiment.." << std::endl;

        if (rounds != 1 && matrix.rows() != matrix.cols())
        {
            std::cerr << "Multi-round was requested, but not supported by matrix." << std::endl;
            return std::nullopt;
        }

        Graph graph = create_graph_add_codelets(device);

        auto tensors = map<string, Tensor>{};
        auto programs = map<string, Program>{};

        auto ipu_matrix = prepare_data(matrix, device.getTarget().getNumTiles());

        std::cerr << "Building programs.." << std::endl;

        build_compute_graph(graph, tensors, programs, device.getTarget().getNumTiles(), ipu_matrix, rounds);
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

        engine.connectStream("toipu_matrix", ipu_matrix.matrix.data());
        engine.connectStream("toipu_idx", ipu_matrix.idx.data());
        engine.connectStream("toipu_row_idx", ipu_matrix.row_idx.data());
        engine.connectStream("toipu_vec", vec.data());

        auto result_vec = vector<float>(ipu_matrix.n);
        engine.connectStream("fromipu_vec", result_vec.data());

        // Run all programs in order
        std::cerr << "Running programs.." << std::endl;
        std::cerr << "Copy data to IPU\n";

        auto copy_timing_start = std::chrono::high_resolution_clock::now();
        engine.run(programIds["copy_to_ipu_matrix"], "copy matrix");
        engine.run(programIds["copy_to_ipu_vec"], "copy vector");
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

        // std::cout << "Resulting vector:\n";
        long int res = 0;
        for (auto v : result_vec)
        {
        //     std::cout << v << ", ";
            res += static_cast<long int>(v);
        }
        // std::cout << std::endl;



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