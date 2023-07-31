# Continues Matrix-vector products on GraphCore's IPU's
This repository contains code to 'quickly' calculate the product of a matrix and vector repeatadly on [Graphcore's IPU's](https://www.graphcore.ai/products/ipu). To run the code you need access to Graphcore's propiatery [Poplar library](https://www.graphcore.ai/products/poplar).

This code was written in connection to my masther thesis at the [University of Bergen](https://uib.no/en) at the [Algorithms department](https://www.uib.no/en/rg/algo). Please do not expect any support whatsover - especially since the author might not have any access anymore to the required hardware and/or software.

## Compilation
The following tooling are required for compilation:
- CMake (build tooling)
- [Conan](https://conan.io/) (C++ dependency management)
- Graphcore SDK (3.0.0)

After downloading the required tooling, choose a suitable build location, e.g. the folder `build`. There the following can be executed:
```sh
$ cmake -DCMAKE_BUILD_TYPE=Debug ..
$ conan install ..
$ make
```
This will generate the build files, download the dependencies and finally build the project. The finished build `matrix-ipu-calc` can be found in the folder `src` in your build folder.

## Running


### IDE setup
The IDE can be difficult due to the various amount of different propriatery tooling needed, a guide (and justification for this work) for the CLion IDE can be found [here](https://github.com/UoB-HPC/ipu-hpc-cookbook/tree/main/productive-dev-workflow). It is most likely only possible to have acces to the necessary libraries on a remote machine. This project was originally developed with the help of VSCode. The following include path was used:
- `${workspaceFolder}/src/**`
- `${workspaceFolder}/include/**`
- `$POPLAR_SDK/lib/graphcore/include/**`: the libraries for IPU code
- `~/.conan/data/**`: Conan dependencies

Known problems:
- The CMake extension for VSCode may overwrite your IntelliSense setup. Since the IPU code is not compiled by CMake the associatied libraries are not included in the include path. Can be fixed by disabling the CMake extension in VSCode. Build scripts can be manually created to invoke CMake.
- The indexing of the libraries might taike painstakingly long. This was fixed by setting `browse.path = []` and `browse.limitSymbolsToIncludedHeaders = true` in the CPP properties.

## Development
Simply said development for the Graphcore IPU is split into two parts: the host code, and vertices which will be run on the IPU in a true MIMD (multiple instructions, multiple data) fashion. The IPU has a limited instruction set and has its own typesystem. The IPU code can be found in the `src/codelets` folder.

The host reads input and compiles a compute graph for the IPU to be executed. 

If you're serious about development for the IPU I recommend the following resources:
- https://github.com/graphcore/tutorials/tree/sdk-release-3.0/tutorials/poplar
- https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/ipu_introduction.html
- https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/
- https://docs.graphcore.ai/projects/poplar-api/en/latest/
- https://github.com/UoB-HPC/ipu-hpc-cookbook

## Thanks
Thank you to my supervisors Fredrik Manne (UiB) and Johannes Langguth (Simula), and to the hardware support at the Simula institute via the [eX3 project](https://www.ex3.simula.no/).