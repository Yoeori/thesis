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
        ("seed", "Seed used when randomness is involved, otherwise `time(NULL)` is used");

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
    Config::get().seed = result.count("seed") ? result["seed"].as<unsigned int>() : time(NULL);
    
    // We read in the matrix
    // FILE* matrix_file = fopen(result["matrix"].as<string>().c_str(), "r");

    // if (matrix_file == nullptr) {
    //     std::cerr << "Something wen't wrong opening the input matrix, exiting" << std::endl;
    //     return EXIT_FAILURE;
    // }

    // std::cout << "Reading matrix..." << std::endl;
    // auto mtx = matrix::read_matrix_market<float>(matrix_file);
    // fclose(matrix_file);

    // auto mtx = optional(matrix::identity<float>(1000));
    // auto mtx = optional(matrix::times<float>(2000, 2.0));
    auto mtx = optional(matrix::ones<float>(7000));
    // std::vector<float> v(1000);
    // std::iota(std::begin(v), std::end(v), 1);
    // auto mtx = optional(matrix::identity_from_iterator(v));

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

    // Perform SpMV experiment
    auto rounds = result["rounds"].as<int>();
    exp_spmv::execute(device.value(), *mtx, rounds);

    // if (Config::get().debug)
    // {
    //     serialize_graph(res.value().graph);
    //     res.value().engine.printProfileSummary(std::cout, OptionFlags{});
    // }

    // std::cout << "Resulting vector:\n";
    // long int res = 0;
    // for (auto v : result_vec)
    // {
    //     // std::cout << v << ", ";
    //     res += static_cast<long int>(v);
    // }
    // // std::cout << std::endl;
    // std::cout << "Sum: " << res << std::endl;
    
    // std::cout << "Sum: " << std::accumulate(result_vec.begin(), result_vec.end(), decltype(result_vec)::value_type(0)) << "\n";

    // Perform calculation locally on CPU, and check results.


    return 0;
}