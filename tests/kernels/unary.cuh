#pragma once

template <typename T, typename F>
__global__ void unary(int N, T* dst, const T* src)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    F f;
    dst[i] = f(src[i]);
}

