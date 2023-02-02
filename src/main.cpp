#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>

#include <cxxopts.hpp>

#include "config.cpp"
#include "ipu.cpp"

#include <poplar/DeviceManager.hpp>

using namespace std;

int main(int argc, char *argv[]) {

    // Define global options
    cxxopts::Options options("matrix-ipu-calc", "Run matrix vector product calculations on the Graphcore IPU");

    options.add_options()
        ("d,debug", "Enable debugging") // Enables debug tooling for the IPU
        ("v,verbose", "Verbose output")
        ("model", "Run on the IPU simulator")
        ("matrix", "The input matrix", cxxopts::value<string>())
        ("h,help", "Print me!");

    options.positional_help("[matrix.mtx]");
    options.parse_positional({"matrix", ""});

    auto result = options.parse(argc, argv);

    if ((!result.count("matrix") && !result.count("model")) || result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    Config::get().debug = result.count("debug");
    Config::get().verbose = result.count("debug");

    // We setup the base IPU functionality:

    // 1. Connect to an IPU
    optional<poplar::Device> device;

    if (result.count("model")) {
        device = getIpuModel(1, 1472);
    } else {
        device = getIpuDevice();
    }

    if (!device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    Graph graph = createGraphAndAddCodelets(device.value());

    // Simple explanation of program flow:
    // 1. Copy data to tiles
    // 2. MatrixBlock calculation (see codelet MatrixBlock.cpp)
    // 3. Copy tile to tile
    // 4. Reduce values (addition) (see codelet Reducer.cpp)
    // 5. Copy reduced value to tiles
    // 6. Repeat 2 until enough iterations are reached.



    // 2. Compile the codelets for the given IPU
    // createGraphAndAddCodelets(device.value());


    // 

    // if (result["debug"].as<bool>()) {
    //     std::cout << "Debugging is on!\n";
    // }


    // FILE* f = fopen("/home/yoeri/data/delaunay_n10.mtx", "r");
    // if (f == nullptr) {
    //     std::cout << "Could not open test file";
    // }

    // open_matrix(f);
    // return 0;

 
}