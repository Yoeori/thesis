#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>

using namespace std;

extern "C" {
    #include "mmio.h" 
}

namespace matrix {
    class Matrix {
    int* m;

        public:
            Matrix() {

            }
    };

    auto read(FILE* f) -> optional<Matrix*> {
        MM_typecode matcode;

        if (mm_read_banner(f, &matcode) != 0) {
            return std::nullopt;
        }

        int M, N, nz;
        if ((mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
            return std::nullopt;
        }

        auto I = (int *) malloc(nz * sizeof(int));
        auto J = (int *) malloc(nz * sizeof(int));
        auto val = (double *) malloc(nz * sizeof(double));

        return std::nullopt;
    };

    auto open(string file) -> optional<Matrix*> {
        FILE* f;

        if ((f = fopen(file.c_str(), "r")) == NULL)
            return std::nullopt;

        return read(f);
    }

}

