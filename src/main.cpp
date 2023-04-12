#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>
#include <numeric>

#include <cxxopts.hpp>

#include "config.cpp"
#include "ipu.cpp"
#include "cpu.cpp"

#include <poplar/DeviceManager.hpp>

using namespace std;

// https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

int main(int argc, char *argv[])
{

    // Define global options
    cxxopts::Options options("matrix-ipu-calc", "Run matrix vector product calculations on the Graphcore IPU");

    options.add_options()("d,debug", "Enable debugging") // Enables debug tooling for the IPU
        ("v,verbose", "Verbose output")("model", "Run on the IPU simulator")("matrix", "The input matrix", cxxopts::value<string>())("h,help", "Print me!");

    options.positional_help("[matrix.mtx]");
    options.parse_positional({"matrix", ""});

    auto result = options.parse(argc, argv);

    if ((!result.count("matrix") && !result.count("model")) || result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    Config::get().debug = result.count("debug");
    Config::get().verbose = result.count("verbose");

    // We read in the matrix
    // FILE* matrix_file = fopen(result["matrix"].as<string>().c_str(), "r");

    // if (matrix_file == nullptr) {
    //     std::cerr << "Something wen't wrong opening the input matrix, exiting" << std::endl;
    //     return EXIT_FAILURE;
    // }

    // std::cout << "Reading matrix..." << std::endl;
    // auto mtx = matrix::read_matrix_market<float>(matrix_file);
    // fclose(matrix_file);

    auto mtx = optional(matrix::times<float>(370, 2));

    if (!mtx.has_value())
    {
        std::cerr << "Something wen't wrong reading the input matrix, exiting" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Finished reading matrix." << std::endl;
    // mtx.value().shuffle();

    // We setup the base IPU functionality:
    // 1. Connect to an IPU
    std::cout << "Setting up IPU device." << std::endl;
    optional<poplar::Device> device;

    if (result.count("model"))
    {
        device = getIpuModel(1, 1472);
    }
    else
    {
        device = getIpuDevice();
    }

    if (!device.has_value())
    {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        std::cerr << "Try starting with --model for a virtual IPU" << std::endl;
        return EXIT_FAILURE;
    }

    Graph graph = create_graph_add_codelets(device.value());

    auto tensors = map<string, Tensor>{};
    auto programs = map<string, Program>{};

    auto ipu_matrix = prepare_data(mtx.value(), device->getTarget().getNumTiles());

    std::cout << "Building programs.." << std::endl;
    auto rounds = result["rounds"].as<int>();
    build_compute_graph(graph, tensors, programs, device->getTarget().getNumTiles(), ipu_matrix, rounds);
    build_data_streams(graph, tensors, programs, ipu_matrix);
    build_full_compute(graph, tensors, programs, ipu_matrix);

    auto ENGINE_OPTIONS = OptionFlags{};

    if (Config::get().debug)
    {
        ENGINE_OPTIONS = OptionFlags{
            {"target.saveArchive", "archive.a"},
            {"debug.instrument", "true"},
            {"debug.instrumentCompute", "true"},
            {"debug.loweredVarDumpFile", "vars.capnp"},
            {"debug.instrumentControlFlow", "true"},
            {"debug.computeInstrumentationLevel", "tile"},
            {"debug.outputAllSymbols", "true"},
            {"autoReport.all", "true"},
            {"autoReport.outputSerializedGraph", "true"},
            {"debug.retainDebugInformation", "true"},
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
    engine.load(*device);

    if (Config::get().debug)
    {
        engine.enableExecutionProfiling();
    }

    auto vec = vector<float>(ipu_matrix.n, 1.0);

    engine.connectStream("toipu_matrix", ipu_matrix.matrix.data());
    engine.connectStream("toipu_idx", ipu_matrix.idx.data());
    engine.connectStream("toipu_row_idx", ipu_matrix.row_idx.data());
    engine.connectStream("toipu_vec", vec.data());

    auto result_vec = vector<float>(ipu_matrix.n);
    engine.connectStream("fromipu_vec", result_vec.data());

    // Run all programs in order
    std::cout << "Running programs.." << std::endl;
    engine.run(programIds["main"], "main program");
    // engine.run(programIds["copy_to_ipu_matrix"], "copy matrix");
    // engine.run(programIds["copy_to_ipu_vec"], "copy vector");

    // std::cout << "Copying done.." << std::endl;

    // auto rounds = result["rounds"].as<int>();    
    // for (auto i = 1; i <= rounds; i++)
    // {
    //     std::cout << "Round " << i << std::endl;
    //     engine.run(programIds["spmv"], string_format("SpMV %i", i));
    //     std::cout << "SPMV done.." << std::endl;

    //     // engine.run(programIds["print_result_vec"], string_format("Print vec %d", i));

    //     engine.run(programIds["reduce"], string_format("Reduce %i", i));
    //     std::cout << "Reduce done.." << std::endl;
    // }

    // engine.run(programIds["copy_to_host"], "copy to host");

    if (Config::get().debug)
    {
        serialize_graph(graph);
        engine.printProfileSummary(std::cout, OptionFlags{});
    }

    for (auto v : result_vec)
        std::cout << v << ", ";
    std::cout << std::endl;

    return 0;
}