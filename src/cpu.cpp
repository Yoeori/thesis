#include <iostream>
#include <cassert>
#include <algorithm>
#include <type_traits>

using namespace std;

namespace cpu
{

    /**
     * Perform one round of matrix vector multiplication
     * This is not meant to be efficient and should mainly be used to check results from other sources
     *
     * @param matrix a matrix given as a one-dimensional array with n * m elements
     * @param vector a vector with m elements
     * @param n number of rows in matrix
     * @param m number of cols in matrix
     * @param res variable where the result will be written to
     */

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    void calc_mv_product(T *matrix, T *vector, int n, int m, T *res)
    {
        for (int row = 0; row < n; row++)
        {
            res[row] = 0;
            for (int col = 0; col < m; col++)
            {
                res[row] += matrix[row * m + col] * vector[col];
            }
        }
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    void multi_round_mv_product(T *matrix, T *vector, int n, int m, int rounds)
    {
        // We require a square matrix, otherwise our vector sizes would differ.
        assert(n == m);

        T *res = new T[n];

        for (int round = 0; round < rounds; round++)
        {
            calc_mv_product(matrix, vector, n, m, res);

            T *tmp = vector;
            vector = res;
            res = tmp;
        }

        if (rounds % 2 != 0)
        {
            copy(vector, vector + n, res);

            T *tmp = vector;
            vector = res;
            res = tmp;
        }

        delete[] res;
    }

    /**
     * Compare two vectors and returns if they're equal
     */
    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    bool check_vector(T *one, T *two, int n)
    {
        for (int i = 0; i < n; i++)
        {
            if (one[i] != two[i])
            {
                return false;
            }
        }
        return true;
    }
}