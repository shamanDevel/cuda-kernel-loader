#pragma once

template <typename T>
__global__ void AxpyKernel(int N, T alpha, const T* x, T* y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    y[i] += alpha * x[i];
}

