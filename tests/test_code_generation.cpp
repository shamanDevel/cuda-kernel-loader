#include "catch.hpp"

#include <ckl/kernel_loader.h>
#include <ckl/errors.h>
#include <cuda_runtime.h>

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(kernels);
#include <ckl/cmrc_loader.h>

TEST_CASE("code_generation", "[ckl]")
{
    cmrc::embedded_filesystem fs = cmrc::kernels::get_filesystem();

    auto kl = std::make_shared<ckl::KernelLoader>();
    kl->disableCudaCache();
    kl->setFileLoader(std::make_shared<ckl::CMRCLoader>(fs));

    //define the custom unary function
    const auto fHost = [](float v) {return v * v; };
    const std::string fDevice = R"code(
struct Functor {
    __device__ float operator()(float v) const {return v*v;}
};
#include "unary.cuh"
    )code";

    const int N = 2048;
    std::vector<float> srcHost(N);
    std::vector<float> dstHost(N);
    for (int i = 0; i < N; ++i) srcHost[i] = i*0.5f;

    float* srcDevice;
    float* dstDevice;
    CKL_SAFE_CALL(cudaMalloc(&srcDevice, sizeof(float) * N));
    CKL_SAFE_CALL(cudaMalloc(&dstDevice, sizeof(float) * N));
    CKL_SAFE_CALL(cudaMemcpy(srcDevice, srcHost.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

    //compute - CPU
    for (int i = 0; i < N; ++i) dstHost[i] = fHost(srcHost[i]);

    //compute - GPU
    auto fun = kl->getKernel(
        "unary<float, Functor>",
        fDevice
    );
    unsigned blockDim = fun->bestBlockSize();
    unsigned gridDim = CKL_DIV_UP(N, blockDim);
    CUstream stream = nullptr;
    fun->call(gridDim, blockDim, 0, stream,
        N, dstDevice, srcDevice);

    //and copy back
    std::vector<float> dstDeviceHost(N);
    CKL_SAFE_CALL(cudaMemcpy(dstDeviceHost.data(), dstDevice, sizeof(float) * N, cudaMemcpyDeviceToHost));

    //compare
    for (int i = 0; i < N; ++i)
    {
        INFO("i: " << i);
        REQUIRE(dstHost[i] == Approx(dstDeviceHost[i]));
    }

    CKL_SAFE_CALL(cudaFree(srcDevice));
    CKL_SAFE_CALL(cudaFree(dstDevice));
}