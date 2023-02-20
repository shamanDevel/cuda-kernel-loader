#include <ckl/errors.h>
#include <spdlog/spdlog.h>

#ifndef CKL_BACKTRACE_ON_ERROR
#define CKL_BACKTRACE_ON_ERROR 1
#endif

CKL_NAMESPACE_BEGIN

static void logError(const char* error, const char* file, const int line)
{
    spdlog::error("CUDA error failed at {}:{} : {}", file, line, error);
#if CKL_BACKTRACE_ON_ERROR == 1
    spdlog::dump_backtrace();
#endif
}

bool internal::ErrorHelpers::evalError(cudaError err, const char* file, const int line)
{
    if (cudaErrorCudartUnloading == err) {
        std::string msg = internal::Format::format("cudaCheckError() failed at %s:%i : %s\nThis error can happen in "
                                                   "multi-threaded applications during shut-down and is ignored.\n",
                                                   file, line, cudaGetErrorString(err));
        // TODO: soft exception?
        return false;
    }
    else if (cudaSuccess != err) {
        logError(cudaGetErrorString(err), file, line);
        std::string msg =
                internal::Format::format("cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        throw cuda_error(msg);
    }
    return true;
}
bool internal::ErrorHelpers::evalErrorNoThrow(cudaError err, const char* file, const int line)
{
    if (cudaSuccess != err) {
        logError(cudaGetErrorString(err), file, line);
        return false;
    }
    return true;
}

bool internal::ErrorHelpers::evalError(CUresult err, const char* file, const int line)
{
    if (CUDA_SUCCESS != err) {
        const char* pStr;
        cuGetErrorString(err, &pStr);
        const char* pName;
        cuGetErrorName(err, &pName);
        std::string msg = internal::Format::format("cudaSafeCall() failed at %s:%i : Error code %s, description: %s\n",
                                                   file, line, pName, pStr);
        logError(pName, file, line);
        throw cuda_error(msg);
    }
    return true;
}
bool internal::ErrorHelpers::evalErrorNoThrow(CUresult err, const char* file, const int line)
{
    if (CUDA_SUCCESS != err) {
        const char* pStr;
        cuGetErrorString(err, &pStr);
        const char* pName;
        cuGetErrorName(err, &pName);
        logError(pName, file, line);
        return false;
    }
    return true;
}

void internal::ErrorHelpers::cudaSafeCall(cudaError err, const char* file, const int line)
{
    if (!evalError(err, file, line)) return;
#if CKL_ALWAYS_SYNC == 1
    // insert a device-sync
    err = cudaDeviceSynchronize();
    evalError(err, file, line);
#endif
}

bool internal::ErrorHelpers::cudaSafeCallNoThrow(cudaError err, const char* file, const int line)
{
    if (!evalErrorNoThrow(err, file, line)) return false;
#if CKL_ALWAYS_SYNC == 1
    // insert a device-sync
    err = cudaDeviceSynchronize();
    if (!evalErrorNoThrow(err, file, line)) return false;
#endif
    return true;
}

void internal::ErrorHelpers::cudaSafeCall(CUresult err, const char* file, const int line)
{
    if (!evalError(err, file, line)) return;
#if CKL_ALWAYS_SYNC == 1
    // insert a device-sync
    err = cudaDeviceSynchronize();
    evalError(err, file, line);
#endif
}

bool internal::ErrorHelpers::cudaSafeCallNoThrow(CUresult err, const char* file, const int line)
{
    if (!evalErrorNoThrow(err, file, line)) return false;
#if CKL_ALWAYS_SYNC == 1
    // insert a device-sync
    err = cudaDeviceSynchronize();
    if (!evalErrorNoThrow(err, file, line)) return false;
#endif
    return true;
}

void internal::ErrorHelpers::cudaCheckError(const char* file, const int line)
{
    cudaError err = cudaGetLastError();
    if (!evalError(err, file, line)) return;

#if CKL_ALWAYS_SYNC == 1
    // More careful checking. However, this will affect performance.
    err = cudaDeviceSynchronize();
    evalError(err, file, line);
#endif
}

CKL_NAMESPACE_END
