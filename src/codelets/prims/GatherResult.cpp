#ifndef __IPU_ARCH_VERSION__
#define __IPU_ARCH_VERSION__ 2
#endif

#include <poplar/Vertex.hpp>
#include <arch/gc_tile_defines.h>

#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>
#include <limits.h>
#include "print.h"

using namespace poplar;

// This class uses as SupervisorVertex to control multiple vertices, instead of using MultiVertex
// MultiVertex does not have a way to collect results from different threads
class GatherResult : public Vertex
{
public:
    Input<Vector<int>> block_dist;
    Input<Vector<unsigned>> block_dist_from;
    Input<Vector<unsigned>> block_dist_to;

    Output<unsigned> current;
    InOut<Vector<unsigned>> connection;

    Output<bool> should_continue;

    Vector<int> tmp1;      // block_dist result for each thread
    Vector<unsigned> tmp2; // block_dist_from result for each thread
    Vector<unsigned> tmp3; // block_dist_to result for each thread

    bool compute()
    {
        unsigned workerId = __builtin_ipu_get(CSR_W_WSR__INDEX) & CSR_W_WSR__CTXTID_M1__MASK;
        unsigned workers = CTXT_WORKERS;

        int best_dist = INT_MAX;
        unsigned from = 0;
        unsigned to = 0;

        for (size_t i = workerId; i < block_dist.size(); i += workers)
        {
            if (block_dist[i] < best_dist)
            {
                best_dist = block_dist[i];
                from = block_dist_from[i];
                to = block_dist_to[i];
            }
        }

        tmp1[workerId] = best_dist;
        tmp2[workerId] = from;
        tmp3[workerId] = to;

        return true;
    }
};

class GatherResultSupervisor : public SupervisorVertex
{
public:
    Input<Vector<int>> block_dist;
    Input<Vector<unsigned>> block_dist_from;
    Input<Vector<unsigned>> block_dist_to;

    Output<unsigned> current;
    InOut<Vector<unsigned>> connection;

    Output<bool> should_continue;

    Vector<int> tmp1;      // block_dist result for each thread
    Vector<unsigned> tmp2; // block_dist_from result for each thread
    Vector<unsigned> tmp3; // block_dist_to result for each thread

    __attribute__((target("supervisor"))) void collect()
    {
        // use tmp to write back out value;
        unsigned res1 = tmp1[0] < tmp1[1] ? 0 : 1;
        unsigned res2 = tmp1[2] < tmp1[3] ? 2 : 3;
        unsigned res3 = tmp1[4] < tmp1[5] ? 4 : 5;

        res1 = tmp1[res1] < tmp1[res2] ? res1 : res2;
        res1 = tmp1[res1] < tmp1[res3] ? res1 : res3;

        if (tmp1[res1] == INT_MAX)
        {
            *should_continue = false;
        }
        else
        {
            *current = tmp3[res1];
            connection[tmp3[res1]] = tmp2[res1];
            *should_continue = true;
        }
    }

    __attribute__((target("supervisor"))) bool compute()
    {
        __asm__ volatile(
            "setzi   $m1, __runCodelet_GatherResult\n"
            "runall  $m1, $m0 , 0 \n"
            "sync   %[sync_zone]\n" ::[sync_zone] "i"(TEXCH_SYNCZONE_LOCAL));

        collect();
        return true;
    }
};