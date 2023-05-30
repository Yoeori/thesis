#ifndef THESIS_MATRIX_HPP
#define THESIS_MATRIX_HPP

#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <map>
#include <random>
#include <assert.h>

#include "config.cpp"

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

    /**
     * Generates a random permutation of a given size which can be apply on any vector or matrix.
     * Includes helper to reverse permutation
     */
    struct Permutation
    {
    public:
        Permutation(size_t size, unsigned int seed) : size(size)
        {
            perm = vector<size_t>(size);
            rev = vector<size_t>(size);

            srand(seed);
            std::iota(perm.begin(), perm.end(), 0);
            std::random_shuffle(perm.begin(), perm.end());

            // Fill rev
            for (size_t i = 0; i < perm.size(); i++)
            {
                rev[perm[i]] = i;
            }
        }

        Permutation(size_t size) : Permutation(size, Config::get().seed)
        {
        }

        size_t apply(size_t in)
        {
            assert(in < size);
            return perm[in];
        }

        size_t reverse(size_t in)
        {
            assert(in < size);
            return rev[in];
        }

    private:
        size_t size;
        vector<size_t> perm;
        vector<size_t> rev;
    };

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    struct Matrix
    {
    public:
        Matrix(int m, int n, int nz) : applied_perm(nullopt), m(m), n(n), nz(nz)
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

        void permutate(Permutation perm)
        {
            // We generate a new matrix
            auto new_matrix = vector<T>(m * n);

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    new_matrix[perm.apply(i) * n + perm.apply(j)] = matrix[i * n + j];
                }
            }

            // Apply permutation
            matrix = new_matrix;
            applied_perm = optional(perm);
        }

        void permutate()
        {
            permutate(Permutation(n));
        }

        optional<Permutation> applied_perm;

    private:
        vector<T> matrix;
        int m;  // rows
        int n;  // cols
        int nz; // non-zero values
    };

    // Structure meant for _very large_ (read: high number of rows/cols) sparse matrixes
    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    struct SparseMatrix
    {
    public:
        SparseMatrix(long m, long n, size_t nz) : applied_perm(nullopt), m(m), n(n), nz(nz)
        {
            i = vector<long>(nz);
            j = vector<long>(nz);
            v = vector<T>(nz);
        }

        void permutate(Permutation perm)
        {
            // We generate a new matrix
            auto new_i = vector<long>(nz);
            auto new_j = vector<long>(nz);

            for (size_t k = 0; k < nz; k++)
            {
                new_i[k] = perm.apply(i[k]);
                new_j[k] = perm.apply(j[k]);
            }

            // Apply permutation
            i = new_i;
            j = new_j;
            applied_perm = optional(perm);
        }

        void permutate()
        {
            permutate(Permutation(n));
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

        void set(size_t idx, long vi, long vj, T value)
        {
            assert(idx < nz);
            i[idx] = vi;
            j[idx] = vj;
            v[idx] = value;
        }

        tuple<long, long, T> get(size_t idx)
        {
            assert(idx < nz);
            return make_tuple(i[idx], j[idx], v[idx]);
        }

        optional<Permutation> applied_perm;

    private:
        vector<T> v;
        vector<long> i;
        vector<long> j;

        long m;
        long n;
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

        if (mm_is_symmetric(matcode))
        {
            nz = nz * 2;
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

            if (mm_is_symmetric(matcode))
            {
                i++;
                matrix.set((jm - 1), (im - 1), val);
            }
        }

        return optional(matrix);
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    optional<SparseMatrix<T>> read_matrix_market_sparse(FILE *f)
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

        if (mm_is_symmetric(matcode))
        {
            nz = nz * 2;
        }

        SparseMatrix<T> matrix(m, n, nz);

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

            matrix.set(i, (im - 1), (jm - 1), val);

            if (mm_is_symmetric(matcode))
            {
                i++;
                matrix.set(i, (jm - 1), (im - 1), val);
            }
        }

        return optional(matrix);
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    Matrix<T> times(int size, T n, int top_offsett)
    {
        Matrix<T> matrix(size, size, size - top_offsett);
        for (auto i = 0; i < size - top_offsett; i++)
        {
            matrix.set(i + top_offsett, i, n);
        }
        return matrix;
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    Matrix<T> times(int size, T n)
    {
        return times(size, n, 0);
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    Matrix<T> identity(int size)
    {
        return times(size, (T)1);
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    Matrix<T> identity(int size, int top_offset)
    {
        return times(size, (T)1, top_offset);
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
        Matrix<T> matrix(size, size, size * size);
        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < size; y++)
            {
                matrix.set(x, y, (T)1);
            }
        }
        return matrix;
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    SparseMatrix<T> sparse_times(int size, T n, int top_offsett)
    {
        SparseMatrix<T> matrix(size, size, size - top_offsett);
        for (auto i = 0; i < size - top_offsett; i++)
        {
            matrix.set(i, i + top_offsett, i, n);
        }
        return matrix;
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    SparseMatrix<T> sparse_times(int size, T n)
    {
        return sparse_times(size, n, 0);
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    SparseMatrix<T> sparse_identity(int size)
    {
        return sparse_times(size, (T)1);
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    SparseMatrix<T> sparse_identity(int size, int top_offset)
    {
        return sparse_times(size, (T)1, top_offset);
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    SparseMatrix<T> sparse_identity_from_iterator(vector<T> vec)
    {
        SparseMatrix<T> matrix(vec.size(), vec.size(), vec.size());
        for (size_t i = 0; i < vec.size(); i++)
        {
            matrix.set(i, i, i, vec[i]);
        }
        return matrix;
    }

}

#endif