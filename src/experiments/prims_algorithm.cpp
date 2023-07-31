#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <float.h>
#include <limits.h>

#include <poplar/Engine.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <popops/codelets.hpp>
#include <poplar/CycleCount.hpp>
#include <poplar/PrintTensor.hpp>

#include "../matrix.hpp"
#include "../config.cpp"
#include "../ipu.cpp"
#include "../report.cpp"

using ::poplar::Device;
using ::poplar::Engine;
using ::poplar::Graph;
using ::poplar::OptionFlags;
using ::poplar::Tensor;
using ::poplar::SyncType;

using ::poplar::program::Program;
using ::poplar::program::Copy;
using ::poplar::program::Sequence;
using ::poplar::program::Execute;
using ::poplar::program::RepeatWhileTrue;

using ::poplar::FLOAT;
using ::poplar::INT;
using ::poplar::UNSIGNED_INT;
using ::poplar::BOOL;

namespace exp_prims
{
    // Private namespace
    namespace
    {
        struct PrimsIPU
        {
            PrimsIPU(vector<int> values, vector<unsigned int> idx_row, vector<unsigned int> idx_col, unsigned int n, vector<size_t> block_indices, vector<size_t> block_row_lt, unsigned int blocks) : values(values), idx_row(idx_row), idx_col(idx_col), n(n), block_indices(block_indices), block_row_lt(block_row_lt), blocks(blocks){};

            vector<int> values;
            vector<unsigned int> idx_row;
            vector<unsigned int> idx_col;
            unsigned int n;

            vector<size_t> block_indices;
            vector<size_t> block_row_lt;
            unsigned int blocks;
        };

        PrimsIPU prepare_data(matrix::SparseMatrix<float> &matrix, const int num_tiles)
        {
            unsigned amount_of_blocks = (unsigned)num_tiles;

            // We are building up a CSC data structure, divided over num_tiles blocks vertically

            // First we need to divide our matrix up over the blocks, so we need to go through all nz-values and make a prefix
            // lookup for the rows. Then we can divide evenly with around matrix.nonzeroes() / num_tiles values per block

            vector<int> rows_size = vector(matrix.rows(), 0);
            for (size_t o = 0; o < (unsigned)matrix.nonzeroes(); o++)
            {
                rows_size[get<0>(matrix.get(o))]++;
            }

            int values_per_block = std::max((unsigned)matrix.nonzeroes() / amount_of_blocks + (matrix.nonzeroes() % amount_of_blocks != 0), (unsigned)1);

            // Our 3 main data structures
            vector<int> ipu_values(matrix.nonzeroes(), 0.0);
            vector<unsigned> ipu_row(matrix.nonzeroes(), 0.0);
            vector<unsigned> ipu_column((matrix.cols() + 1) * amount_of_blocks, 0.0);

            // pointer structure for where each block lies in values & row
            vector<size_t> blocks(amount_of_blocks + 1, 0);
            vector<size_t> block_row_lt(amount_of_blocks + 1, 0);
            vector<size_t> block_lt(matrix.rows(), 0);

            // Now we divide up the rows
            // Logic: we count non-zero values until it's equal or above the tresshold (block_cursor * values_per_block)
            unsigned int block_cursor = 0;
            unsigned int total = 0;
            for (size_t row = 0; row < (unsigned)matrix.rows(); row++)
            {
                total += rows_size[row];
                block_lt[row] = block_cursor;

                if (total >= (block_cursor + 1) * values_per_block)
                {
                    blocks[block_cursor + 1] = total;
                    block_row_lt[block_cursor + 1] = row + 1;
                    block_cursor++;
                }
            }

            for (; block_cursor < (unsigned)blocks.size() - 1; block_cursor++)
            {
                blocks[block_cursor + 1] = total;
                block_row_lt[block_cursor + 1] = matrix.rows();
            }

            // Now we can start calculating ipu_column
            for (size_t o = 0; o < (unsigned)matrix.nonzeroes(); o++)
            {
                auto [i, j, v] = matrix.get(o);
                (void)v;

                auto block = block_lt[i];

                ipu_column[(block * (matrix.cols() + 1)) + j + 1]++;
            }

            // stride ipu_col for each block
            for (size_t block = 0; block < amount_of_blocks; block++)
            {
                for (size_t idx = 2; idx < (unsigned)matrix.cols(); idx++)
                {
                    ipu_column[(block * (matrix.cols() + 1)) + idx] += ipu_column[(block * (matrix.cols() + 1)) + idx - 1];
                }
            }

            // populate ipu_values / ipu_row
            vector<unsigned> col_cursor(ipu_column);

            for (size_t o = 0; o < (unsigned)matrix.nonzeroes(); o++)
            {
                auto [i, j, v] = matrix.get(o);
                auto block = block_lt[i];

                auto offset = blocks[block] + col_cursor[(block * (matrix.cols() + 1)) + j];
                ipu_values[offset] = (int)v;
                ipu_row[offset] = i - block_row_lt[block];

                col_cursor[(block * (matrix.cols() + 1)) + j]++;
            }

            return PrimsIPU(ipu_values, ipu_row, ipu_column, matrix.cols(), blocks, block_row_lt, amount_of_blocks);
        }

        auto create_graph_add_codelets(const Device &device) -> Graph
        {
            auto graph = poplar::Graph(device.getTarget());

            // Add our own codelets
            graph.addCodelets({"codelets/prims/PrimsBlock.cpp", "codelets/prims/ReduceBlock.cpp", "codelets/prims/GatherResult.cpp"}, "-I codelets -O3");
            popops::addCodelets(graph);

            return graph;
        }

        void build_compute_graph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, const int num_tiles, PrimsIPU &ipu_matrix, const int loops)
        {
            // The algorithm step by step:
            // 1. we select a vertex (initial 0, current is the previously added vertex)
            // 2. we update dist/dist_prev with that specific vertex (computeset update_dist)
            // 2.1 min of dist to that vertex and current dist
            // 2.2 or remove of current vertex
            // 3. we select the minimal dist (computeset reduce_dist + single reduction over the result)
            // 4. update connection for that vertex and repeat with new vertex (computeset update)

            // Static matrix data
            tensors["weights"] = graph.addVariable(INT, {ipu_matrix.values.size()});
            tensors["idx_row"] = graph.addVariable(UNSIGNED_INT, {ipu_matrix.idx_row.size()});
            tensors["idx_col"] = graph.addVariable(UNSIGNED_INT, {ipu_matrix.blocks, ipu_matrix.n + 1});

            // Result structs
            tensors["connection"] = graph.addVariable(UNSIGNED_INT, {ipu_matrix.n});
            graph.setInitialValue(tensors["connection"][0], 0);

            tensors["dist"] = graph.addVariable(INT, {ipu_matrix.n});
            graph.setInitialValue(tensors["dist"].slice(0, ipu_matrix.n), poplar::ArrayRef(vector<int>(ipu_matrix.n, INT_MAX)));
            tensors["dist_prev"] = graph.addVariable(UNSIGNED_INT, {ipu_matrix.n});

            // Sub-results
            // We store results in dist then reduce for each block, finally find min in reduction (single-threaded)
            tensors["block_dist"] = graph.addVariable(INT, {ipu_matrix.blocks});
            tensors["block_dist_from"] = graph.addVariable(UNSIGNED_INT, {ipu_matrix.blocks});
            tensors["block_dist_to"] = graph.addVariable(UNSIGNED_INT, {ipu_matrix.blocks});

            // Cursor(s)
            tensors["current"] = graph.addVariable(UNSIGNED_INT, {1});
            graph.setInitialValue(tensors["current"][0], 0);

            // Loop variable
            tensors["should_continue"] = graph.addVariable(BOOL, {1});
            graph.setInitialValue(tensors["should_continue"][0], 1);

            auto update_d_cs = graph.addComputeSet("update_dist");

            for (unsigned int block = 0; block < ipu_matrix.blocks; block++)
            {
                auto v = graph.addVertex(update_d_cs, "PrimsBlock", {
                    {"weights", tensors["weights"].slice(ipu_matrix.block_indices[block], ipu_matrix.block_indices[block + 1])}, 
                    {"rows", tensors["idx_row"].slice(ipu_matrix.block_indices[block], ipu_matrix.block_indices[block + 1])}, 
                    {"columns", tensors["idx_col"][block]},
                    {"dist", tensors["dist"].slice(ipu_matrix.block_row_lt[block], ipu_matrix.block_row_lt[block + 1])},
                    {"dist_prev", tensors["dist_prev"].slice(ipu_matrix.block_row_lt[block], ipu_matrix.block_row_lt[block + 1])},
                    {"current", tensors["current"][0]}
                });

                graph.setTileMapping(tensors["weights"].slice(ipu_matrix.block_indices[block], ipu_matrix.block_indices[block + 1]), block);
                graph.setTileMapping(tensors["idx_row"].slice(ipu_matrix.block_indices[block], ipu_matrix.block_indices[block + 1]), block);
                graph.setTileMapping(tensors["idx_col"][block], block);

                graph.setTileMapping(tensors["dist"].slice(ipu_matrix.block_row_lt[block], ipu_matrix.block_row_lt[block + 1]), block);
                graph.setTileMapping(tensors["dist_prev"].slice(ipu_matrix.block_row_lt[block], ipu_matrix.block_row_lt[block + 1]), block);

                graph.setInitialValue(v["row_offset"], ipu_matrix.block_row_lt[block]);

                graph.setPerfEstimate(v, 100); // Needed for simulator
                graph.setTileMapping(v, block);
            }

            auto program_update_d = Execute(update_d_cs);

            auto reduce_d_cs = graph.addComputeSet("reduce_dist");

            for (unsigned int block = 0; block < ipu_matrix.blocks; block++)
            {
                auto v = graph.addVertex(reduce_d_cs, "ReduceBlockSupervisor", {
                    {"dist", tensors["dist"].slice(ipu_matrix.block_row_lt[block], ipu_matrix.block_row_lt[block + 1])},
                    {"dist_prev", tensors["dist_prev"].slice(ipu_matrix.block_row_lt[block], ipu_matrix.block_row_lt[block + 1])},
                    {"block_dist", tensors["block_dist"][block]},
                    {"block_dist_from", tensors["block_dist_from"][block]},
                    {"block_dist_to", tensors["block_dist_to"][block]},
                });

                graph.setFieldSize(v["tmp1"], 6);
                graph.setFieldSize(v["tmp2"], 6);
                graph.setFieldSize(v["tmp3"], 6);

                graph.setPerfEstimate(v, 100); // Needed for simulator
                graph.setTileMapping(v, block);

                graph.setTileMapping(tensors["block_dist"][block], block);
                graph.setTileMapping(tensors["block_dist_from"][block], block);
                graph.setTileMapping(tensors["block_dist_to"][block], block);

                graph.setInitialValue(v["row_offset"], ipu_matrix.block_row_lt[block]);
            }

            auto program_reduce_d = Execute(reduce_d_cs);

            auto gather_result_cs = graph.addComputeSet("gather_results");
            auto v = graph.addVertex(gather_result_cs, "GatherResultSupervisor", {
                {"block_dist", tensors["block_dist"]},
                {"block_dist_from", tensors["block_dist_from"]},
                {"block_dist_to", tensors["block_dist_to"]},
                {"current", tensors["current"][0]},
                {"connection", tensors["connection"]},
                {"should_continue", tensors["should_continue"][0]}
            });

            graph.setPerfEstimate(v, 100); // Needed for simulator
            graph.setTileMapping(v, ipu_matrix.blocks >> 1);

            graph.setTileMapping(tensors["should_continue"][0], ipu_matrix.blocks >> 1);
            graph.setTileMapping(tensors["current"][0], ipu_matrix.blocks >> 1);
            graph.setTileMapping(tensors["connection"], ipu_matrix.blocks >> 1);

            graph.setFieldSize(v["tmp1"], 6);
            graph.setFieldSize(v["tmp2"], 6);
            graph.setFieldSize(v["tmp3"], 6);

            auto program_gather_result = Execute(gather_result_cs);

            auto main_sequence = Sequence{RepeatWhileTrue(Sequence{}, tensors["should_continue"][0], Sequence{program_update_d, program_reduce_d, program_gather_result})};

            if (!Config::get().model)
            {
                auto timing = poplar::cycleCount(graph, main_sequence, 0, SyncType::INTERNAL, "timer");
                graph.createHostRead("readTimer", timing, true);
            }

            programs["main"] = main_sequence;
        }

        auto build_data_streams(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, PrimsIPU &ipu_matrix)
        {
            auto toipu_weights = graph.addHostToDeviceFIFO("toipu_weights", INT, ipu_matrix.values.size());
            auto toipu_idx_row = graph.addHostToDeviceFIFO("toipu_idx_row", UNSIGNED_INT, ipu_matrix.idx_row.size());
            auto toipu_idx_col = graph.addHostToDeviceFIFO("toipu_idx_col", UNSIGNED_INT, ipu_matrix.idx_col.size());

            auto fromipu_connection = graph.addDeviceToHostFIFO("fromipu_connection", UNSIGNED_INT, ipu_matrix.n);

            auto copyto_weights = Copy(toipu_weights, tensors["weights"]);
            auto copyto_idx_row = Copy(toipu_idx_row, tensors["idx_row"]);
            auto copyto_idx_col = Copy(toipu_idx_col, tensors["idx_col"]);

            auto copyhost_connection = Copy(tensors["connection"], fromipu_connection);

            programs["copy_to_ipu_matrix"] = Sequence{copyto_weights, copyto_idx_row, copyto_idx_col};
            programs["copy_to_host"] = copyhost_connection;
        }

    }

    optional<ExperimentReportIPU> execute(const Device &device, matrix::SparseMatrix<float> &matrix, int rounds)
    {
        std::cerr << "Executing Prims experiment.." << std::endl;

        if (Config::get().model)
        {
            std::cerr << "Using the simulator is not supported by this experiment" << std::endl;
            return std::nullopt;
        }

        auto ipu_matrix = prepare_data(matrix, device.getTarget().getNumTiles());

        auto graph = create_graph_add_codelets(device);
        auto tensors = map<string, Tensor>{};
        auto programs = map<string, Program>{};

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

        engine.connectStream("toipu_weights", ipu_matrix.values.data());
        engine.connectStream("toipu_idx_row", ipu_matrix.idx_row.data());
        engine.connectStream("toipu_idx_col", ipu_matrix.idx_col.data());

        auto result_connection_vec = vector<unsigned>(ipu_matrix.n);
        engine.connectStream("fromipu_connection", result_connection_vec.data());

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
        }

        std::cerr << "Copying back result\n"; 

        auto copyback_timing_start = std::chrono::high_resolution_clock::now();
        engine.run(programIds["copy_to_host"], "copy result");
        auto copyback_timing_end = std::chrono::high_resolution_clock::now();
        auto copyback_timing = std::chrono::duration_cast<std::chrono::nanoseconds>(copyback_timing_end - copyback_timing_start).count() / 1e3;

        // Create report
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