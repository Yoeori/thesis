#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>
#include <numeric>
#include <tuple>

#include <cxxopts.hpp>

#include "config.cpp"
#include "ipu.cpp"
#include "cpu.cpp"
#include "report.cpp"

#include "experiments/sparse_matrix_vector_mult.cpp"
#include "experiments/breadth_first_search.cpp"

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

optional<ExperimentReportIPU> execute_experiment(const poplar::Device & device, string exp, matrix::SparseMatrix<float> matrix, int rounds)
{
    if (exp == "bfs")
    {
        return exp_spmv::execute(device, matrix, rounds);
    } 
    else if (exp == "prims")
    {
        return exp_spmv::execute(device, matrix, rounds);
    }
    else if (exp == "spmv")
    {
        return exp_spmv::execute(device, matrix, rounds);
    }

    throw "Unavailable experiment called";
}

int main(int argc, char *argv[])
{

    // Define global options
    cxxopts::Options options("matrix-ipu-calc", "Run matrix vector product calculations on the Graphcore IPU");

    options.add_options()("d,debug", "Enable debugging") // Enables debug tooling for the IPU
        ("v,verbose", "Verbose output")
        ("model", "Run on the IPU simulator")
        ("matrix", "The input matrix", cxxopts::value<string>())
        ("h,help", "Print me!")
        ("r,rounds", "Amount of SpMV rounds using the previous result", cxxopts::value<int>()->default_value("1"))
        ("seed", "Seed used when randomness is involved, otherwise `time(NULL)` is used", cxxopts::value<unsigned int>())
        ("permutate", "The rows and columns in the matrix are randomized for even work distribution", cxxopts::value<bool>()->default_value("true"))
        ("own-reducer", "Use own reducer", cxxopts::value<bool>()->default_value("false"))
        ("e,experiment", "The experiment to run, options: spmv, bfs, prims", cxxopts::value<string>()->default_value("spmv"));

    options.positional_help("[matrix.mtx]");
    options.parse_positional({"matrix", ""});

    auto result = options.parse(argc, argv);

    if ((!result.count("matrix") && !result.count("model")) || result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    auto experiment = result["experiment"].as<string>();
    if (experiment != "spmv" && experiment != "bfs" && experiment != "prims")
    {
        std::cout << "No valid options for experiment were given (" << experiment << "), please use one of the following: spmv, bfs, prims" << std::endl;
        return EXIT_FAILURE;
    }

    Config::get().debug = result.count("debug");
    Config::get().verbose = result.count("verbose");
    Config::get().seed = result.count("seed") ? result["seed"].as<unsigned int>() : time(NULL);
    Config::get().permutate = result["permutate"].as<bool>();
    Config::get().own_reducer = result["own-reducer"].as<bool>();
    Config::get().model = result.count("model");

    // We read in the matrix
    FILE* matrix_file = fopen(result["matrix"].as<string>().c_str(), "r");

    if (matrix_file == nullptr) {
        std::cerr << "Something wen't wrong opening the input matrix, exiting" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Reading matrix..." << std::endl;
    auto mtx = matrix::read_matrix_market_sparse<float>(matrix_file);
    fclose(matrix_file);

    // auto mtx = optional(matrix::identity<float>(1000, 1));
    // auto mtx = optional(matrix::times<float>(2000, 2.0));
    // auto mtx = optional(matrix::ones<float>(7000));
    // std::vector<float> v(1000);
    // std::iota(std::begin(v), std::end(v), 1);
    // auto mtx = optional(matrix::identity_from_iterator(v));

    if (!mtx.has_value())
    {
        std::cerr << "Something wen't wrong reading the input matrix, exiting" << std::endl;
        return EXIT_FAILURE;
    }

    if (Config::get().permutate)
    {
        std::cout << "Permutating matrix" << std::endl;
        (*mtx).permutate(); // The applied permutation is stored in the matrix
    }

    std::cout << "Finished reading matrix." << std::endl;

    // We setup the base IPU functionality:
    // Connect to an IPU
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

    // Perform experiment
    optional<ExperimentReportIPU> exp_result = execute_experiment(*device, experiment, *mtx, result["rounds"].as<int>());

    if (!exp_result.has_value())
    {
        std::cout << "Something went wrong during execution of the experiment" << std::endl;
        return EXIT_FAILURE;
    }

    if (Config::get().debug)
    {
        serialize_graph(exp_result.value().graph);
        exp_result.value().engine.printProfileSummary(std::cout, OptionFlags{});
        std::cout << exp_result.value().to_json().dump() << std::endl;
    }

    return 0;
}