#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>

using namespace std;

extern "C"
{
#include "mmio.h" // Matrix-market IO library
}

namespace matrix
{
    // Helper lib for reading from fscan with template type.
    namespace fscanf_t
    {
        template <typename T>
        bool read(FILE *f, T &ref)
        {
            return false;
        }

        template <>
        bool read(FILE *f, int &ref)
        {
            fscanf(f, "%i", &ref);
            return true;
        }

        template <>
        bool read(FILE *f, float &ref)
        {
            fscanf(f, "%f", &ref);
            return true;
        }

        template <>
        bool read(FILE *f, long int &ref)
        {
            fscanf(f, "%li", &ref);
            return true;
        }

        template <>
        bool read(FILE *f, double &ref)
        {
            fscanf(f, "%lg", &ref);
            return true;
        }
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    struct Matrix
    {
    public:
        Matrix(int m, int n, int nz) : m(m), n(n), nz(nz)
        {
            matrix = vector<T>(m * n);
        }

        void set(int i, int j, T v)
        {
            matrix[i * n + j] = v;
        }

        T get(int i, int j)
        {
            return matrix[i * n + j];
        }

        int rows()
        {
            return m;
        }

        int cols()
        {
            return n;
        }

        int nonzeroes()
        {
            return nz;
        }

        // Shuffles the Matrix rows and columns uniformly, returns a mapping how the matrix was shuffled
        std::pair<vector<int>, vector<int>> shuffle()
        {
            // General functioning: we first create a mapping, create a set with todo, pop from set and follow chain using a temporary vec
            vector<T> temp(std::max(m, n));

            std::vector<int> row_map(m);
            std::iota(std::begin(row_map), std::end(row_map), 0);
            std::random_shuffle(row_map.begin(), row_map.end());

            std::vector<int> col_map(n);
            std::iota(std::begin(col_map), std::end(col_map), 0);
            std::random_shuffle(col_map.begin(), col_map.end());

            return make_pair(row_map, col_map);
        }

    private:
        vector<T> matrix;
        int m;  // rows
        int n;  // cols
        int nz; // non-zero values
    };

    // Structure meant for _very large_ sparse matrixes
    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    struct SparseMatrix
    {
    public:
        SparseMatrix(long long m, long long n, size_t nz) : m(m), n(n), nz(nz)
        {
            i = vector<long long>(nz);
            j = vector<long long>(nz);
            v = vector<T>(nz);
        }

    private:
        vector<T> v;
        vector<long long> i;
        vector<long long> j;

        long long m;
        long long n;
        size_t nz; // Our matrix is limited by the theoretical max size of a vector
    };

    /**
     * Reads a Matrix-Market file (mainly used for sparse-matrixes) to Matrix class, outputs nullopt on read-error
     */
    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    optional<Matrix<T>> read_matrix_market(FILE *f)
    {
        MM_typecode matcode;

        // Incase of errors we return empty
        // We check if the request matches the data
        if (mm_read_banner(f, &matcode) != 0 || !mm_is_coordinate(matcode))
        {
            return std::nullopt;
        }

        int m, n, nz; // rows, cols, non-zero's
        if ((mm_read_mtx_crd_size(f, &m, &n, &nz)) != 0)
        {
            return std::nullopt;
        }

        Matrix<T> matrix(m, n, nz);

        for (auto i = 0; i < nz; i++)
        {
            int im, jm;
            T val;

            fscanf(f, "%d %d", &im, &jm);
            if (mm_is_pattern(matcode))
            {
                val = 1;
            }
            else if (!fscanf_t::read(f, val))
            {
                return std::nullopt;
            }
            fscanf(f, "\n");

            matrix.set((im - 1), (jm - 1), val);
        }

        return optional(matrix);
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    Matrix<T> times(int size, T n)
    {
        Matrix<T> matrix(size, size, size);
        for (auto i = 0; i < size; i++)
        {
            matrix.set(i, i, n);
        }
        return matrix;
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    Matrix<T> identity(int size)
    {
        return times(size, (T)1);
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    Matrix<T> identity_from_iterator(vector<T> vec)
    {
        Matrix<T> matrix(vec.size(), vec.size(), vec.size());
        for (size_t i = 0; i < vec.size(); i++)
        {
            matrix.set(i, i, vec[i]);
        }
        return matrix;
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    Matrix<T> ones(int size)
    {
        Matrix<T> matrix(size, size, size*size);
        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < size; y++)
            {
                matrix.set(x, y, (T)1);
            }
        }
        return matrix;
    }
}
