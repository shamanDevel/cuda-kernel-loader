#include "catch.hpp"

#include <ckl/kernel_loader.h>
#include <ckl/errors.h>
#include <cuda_runtime.h>

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(kernels);
#include <ckl/cmrc_loader.h>

static void testAXPY(ckl::KernelLoader_ptr kl)
{
    const int N = 2048;
    const int alpha = -2;

    //allocate memory
    std::vector<int> xHost(N);
    std::vector<int> yHost(N);
    for (int i=0; i<N; ++i)
    {
        xHost[i] = i % 11;
        yHost[i] = i % 17;
    }

    int* xDevice;
    int* yDevice;
    CKL_SAFE_CALL(cudaMalloc(&xDevice, sizeof(int) * N));
    CKL_SAFE_CALL(cudaMalloc(&yDevice, sizeof(int) * N));
    CKL_SAFE_CALL(cudaMemcpy(xDevice, xHost.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
    CKL_SAFE_CALL(cudaMemcpy(yDevice, yHost.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

    //compute - CPU
    std::vector<int> resultHost(N);
    for (int i=0; i<N; ++i)
    {
        resultHost[i] = yHost[i] + alpha * xHost[i];
    }

    //compute - GPU
    auto fun = kl->getKernel(
        "AxpyKernel<int>",
        ckl::KernelLoader::MainFile("axpy.cuh")
    );
    unsigned blockDim = fun->bestBlockSize();
    unsigned gridDim = CKL_DIV_UP(N, blockDim);
    CUstream stream = nullptr;
    fun->call(gridDim, blockDim, 0, stream,
        N, alpha, xDevice, yDevice);

    //and copy back
    std::vector<int> resultDevice(N);
    CKL_SAFE_CALL(cudaMemcpy(resultDevice.data(), yDevice, sizeof(int) * N, cudaMemcpyDeviceToHost));

    //compare
    for (int i=0; i<N; ++i)
    {
        INFO("i: " << i);
        REQUIRE(resultHost[i] == resultDevice[i]);
    }

    CKL_SAFE_CALL(cudaFree(xDevice));
    CKL_SAFE_CALL(cudaFree(yDevice));
}

TEST_CASE("filesystem_loader", "[ckl]")
{
    auto currentFile = std::filesystem::path(__FILE__);
    auto kernelDir = currentFile.parent_path() / "kernels";

    auto kl = std::make_shared<ckl::KernelLoader>();
    kl->disableCudaCache();
    kl->setFileLoader(std::make_shared<ckl::FilesystemLoader>(kernelDir));

    testAXPY(kl);
}

TEST_CASE("embedded_loader", "[ckl]")
{
    cmrc::embedded_filesystem fs = cmrc::kernels::get_filesystem();

    auto kl = std::make_shared<ckl::KernelLoader>();
    kl->disableCudaCache();
    kl->setFileLoader(std::make_shared<ckl::CMRCLoader>(fs));

    testAXPY(kl);
}

