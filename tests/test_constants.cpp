#include "catch.hpp"

#include <ckl/kernel_loader.h>
#include <ckl/errors.h>
#include <cuda_runtime.h>
#include <random>

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(kernels);
#include <ckl/cmrc_loader.h>

#define NO_KERNEL
#include "kernels/constants.cuh"

TEST_CASE("constants", "[ckl]")
{
    cmrc::embedded_filesystem fs = cmrc::kernels::get_filesystem();

    auto kl = std::make_shared<ckl::KernelLoader>();
    kl->disableCudaCache();
    kl->setFileLoader(std::make_shared<ckl::CMRCLoader>(fs));

    //config
    const int N = 2048;
    const float step = 3.14f / N;
    Data data;
    std::default_random_engine rng(42);
    std::uniform_real_distribution<float> distr;
    for (int i=0; i<NUM_FREQUENCIES; ++i)
    {
        data.scale[i] = distr(rng);
        data.phase[i] = distr(rng);
    }

    //compute - CPU
    std::vector<float> dstHostEvaluation(N);
    for (int i = 0; i < N; ++i) {
        dstHostEvaluation[i] = evaluate(i * step, data);
    }

    //compute - GPU
    float* dstDevice;
    CKL_SAFE_CALL(cudaMalloc(&dstDevice, sizeof(float) * N));
    auto fun = kl->getKernel(
        "ConstantKernel",
        R"code(
#define DEFINE_KERNEL
#include "constants.cuh"
    )code",
        {"c_data"}
    );
    unsigned blockDim = fun->bestBlockSize();
    unsigned gridDim = CKL_DIV_UP(N, blockDim);
    CUstream stream = nullptr;
    fun->fillConstantMemorySync("c_data", data);
    fun->call(gridDim, blockDim, 0, stream,
        N, dstDevice);
    std::vector<float> dstDeviceEvaluation(N);
    CKL_SAFE_CALL(cudaMemcpy(dstDeviceEvaluation.data(), dstDevice, sizeof(float) * N, cudaMemcpyDeviceToHost));

    //compare
    for (int i = 0; i < N; ++i)
    {
        INFO("i: " << i);
        REQUIRE(dstHostEvaluation[i] == Approx(dstDeviceEvaluation[i]));
    }

    CKL_SAFE_CALL(cudaFree(dstDevice));
}