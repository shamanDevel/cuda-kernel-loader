#pragma once

/**
 * Test kernel for specifying constant structures.
 * For the host code, #define NO_KERNEL.
 */

#ifdef NO_KERNEL
#include "math.h"
#endif

struct Data
{
#define NUM_FREQUENCIES 8
    float scale[NUM_FREQUENCIES];
    float phase[NUM_FREQUENCIES];
};

#ifndef NO_KERNEL
__device__
#endif
inline float evaluate(float t, const Data& data)
{
    //simple combination of sine
    float v = 0.f;
    for (int i=0; i<NUM_FREQUENCIES; ++i)
    {
        v += sinf(data.scale[i] * (t + data.phase[i]));
    }
    return v;
}

#ifndef NO_KERNEL
__constant__ Data c_data;

__global__ void ConstantKernel(int N, float* dst)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    float step = 3.14f / N;
    dst[i] = evaluate(i * step, c_data);
}
#endif
