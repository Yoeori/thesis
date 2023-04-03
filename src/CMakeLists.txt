add_executable(matrix-ipu-calc main.cpp)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# I don't know why this works, I have _no_ idea how CMake works, but it works.
# Pls don't touch
add_library(mmio ../include/mmio.c)
target_include_directories(mmio PUBLIC ../include/)

target_link_libraries(matrix-ipu-calc
        poplar
        poputil
        popops
        mmio
        )

configure_file(codelets/MatrixBlock.cpp codelets/MatrixBlock.cpp COPYONLY)
configure_file(codelets/Reducer.cpp codelets/Reducer.cpp COPYONLY)
configure_file(codelets/ReducerToVector.cpp codelets/ReducerToVector.cpp COPYONLY)