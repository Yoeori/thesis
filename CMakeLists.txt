cmake_minimum_required(VERSION 3.10)
project(MatrixIPU)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -ansi -Wno-deprecated")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Wall -ansi -Wno-deprecated -march=native -mtune=native -O3")
endif()

# Can be useful for debugging but will make Poplar segmentationfault.
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g3 -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC")

find_package(poplar REQUIRED)

add_executable(matrix-ipu-calc src/main.cpp)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_library(mmio include/mmio.c)
target_include_directories(mmio PUBLIC include/)

target_link_libraries(matrix-ipu-calc
        poplar
        poputil
        popops
        mmio
        )

configure_file(src/codelets/spmv/MatrixBlock.cpp codelets/spmv/MatrixBlock.cpp COPYONLY)
configure_file(src/codelets/spmv/ReducerToVector.cpp codelets/spmv/ReducerToVector.cpp COPYONLY)

configure_file(src/codelets/bfs/MatrixBlockBFS.cpp codelets/bfs/MatrixBlockBFS.cpp COPYONLY)
configure_file(src/codelets/bfs/ReducerBFS.cpp codelets/bfs/ReducerBFS.cpp COPYONLY)

# include_directories(include)
# add_subdirectory(src)