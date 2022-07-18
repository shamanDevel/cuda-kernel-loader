#ifndef __CKL_ERRORS_H__
#define __CKL_ERRORS_H__

/**
 * Originally copied from
 * https://gitlab.com/shaman42/cuMat/-/blob/master/cuMat/src/Errors.h
 * (my own source code, I can use it ^^ )
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <exception>
#include <string>
#include <cstdarg>
#include <vector>
#include <stdexcept>
#include <stdio.h>

#include "common.h"

CKL_NAMESPACE_BEGIN

namespace internal {
	struct Format
	{
		static std::string vformat(const char* fmt, va_list ap)
		{
			// Allocate a buffer on the stack that's big enough for us almost
			// all the time.  Be prepared to allocate dynamically if it doesn't fit.
			size_t size = 1024;
			char stackbuf[1024];
			std::vector<char> dynamicbuf;
			char* buf = &stackbuf[0];
			va_list ap_copy;

			while (1) {
				// Try to vsnprintf into our buffer.
				va_copy(ap_copy, ap);
				int needed = vsnprintf(buf, size, fmt, ap);
				va_end(ap_copy);

				// NB. C99 (which modern Linux and OS X follow) says vsnprintf
				// failure returns the length it would have needed.  But older
				// glibc and current Windows return -1 for failure, i.e., not
				// telling us how much was needed.

				if (needed <= (int)size && needed >= 0) {
					// It fit fine so we're done.
					return std::string(buf, (size_t)needed);
				}

				// vsnprintf reported that it wanted to write more characters
				// than we allotted.  So try again using a dynamic buffer.  This
				// doesn't happen very often if we chose our initial size well.
				size = (needed > 0) ? (needed + 1) : (size * 2);
				dynamicbuf.resize(size);
				buf = &dynamicbuf[0];
			}
		}
		//Taken from https://stackoverflow.com/a/69911/4053176

		static std::string format(const char* fmt, ...)
		{
			va_list ap;
			va_start(ap, fmt);
			std::string buf = vformat(fmt, ap);
			va_end(ap);
			return buf;
		}
		//Taken from https://stackoverflow.com/a/69911/4053176
	};
}

class cuda_error : public std::exception
{
private:
	std::string message_;
public:
	cuda_error(std::string message)
		: message_(message)
	{}

	cuda_error(const char* fmt, ...)
	{
		va_list ap;
		va_start(ap, fmt);
		message_ = internal::Format::vformat(fmt, ap);
		va_end(ap);
	}

	const char* what() const throw() override
	{
		return message_.c_str();
	}
};

namespace internal
{
	struct ErrorHelpers
	{

		// Taken from https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
		// and adopted

	private:
		static bool evalError(cudaError err, const char* file, const int line)
		{
			if (cudaErrorCudartUnloading == err) {
				std::string msg = internal::Format::format("cudaCheckError() failed at %s:%i : %s\nThis error can happen in multi-threaded applications during shut-down and is ignored.\n",
					file, line, cudaGetErrorString(err));
				//TODO: soft exception?
				return false;
			}
			else if (cudaSuccess != err) {
				std::string msg = internal::Format::format("cudaSafeCall() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString(err));
				throw cuda_error(msg);
			}
			return true;
		}
		static bool evalErrorNoThrow(cudaError err, const char* file, const int line)
		{
			if (cudaSuccess != err) {
				std::string msg = internal::Format::format("cudaSafeCall() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString(err));
				//TODO: logging?
				return false;
			}
			return true;
		}

		static bool evalError(CUresult err, const char* file, const int line)
		{
			if (CUDA_SUCCESS != err) {
				const char* pStr;
				cuGetErrorString(err, &pStr);
				const char* pName;
				cuGetErrorName(err, &pName);
				std::string msg = internal::Format::format("cudaSafeCall() failed at %s:%i : Error code %s, description: %s\n",
					file, line, pName, pStr);
				throw cuda_error(msg);
			}
			return true;
		}
		static bool evalErrorNoThrow(CUresult err, const char* file, const int line)
		{
			if (CUDA_SUCCESS != err) {
				const char* pStr;
				cuGetErrorString(err, &pStr);
				const char* pName;
				cuGetErrorName(err, &pName);
				std::string msg = internal::Format::format("cudaSafeCall() failed at %s:%i : Error code %s, description: %s\n",
					file, line, pName, pStr);
				//TODO: logging?
				return false;
			}
			return true;
		}
	public:
		static void cudaSafeCall(cudaError err, const char* file, const int line)
		{
			if (!evalError(err, file, line)) return;
#if CKL_ALWAYS_SYNC==1
			//insert a device-sync
			err = cudaDeviceSynchronize();
			evalError(err, file, line);
#endif
		}

		static bool cudaSafeCallNoThrow(cudaError err, const char* file, const int line)
		{
			if (!evalErrorNoThrow(err, file, line)) return false;
#if CKL_ALWAYS_SYNC==1
			//insert a device-sync
			err = cudaDeviceSynchronize();
			if (!evalErrorNoThrow(err, file, line)) return false;
#endif
			return true;
		}

		static void cudaSafeCall(CUresult err, const char* file, const int line)
		{
			if (!evalError(err, file, line)) return;
#if CKL_ALWAYS_SYNC==1
			//insert a device-sync
			err = cudaDeviceSynchronize();
			evalError(err, file, line);
#endif
		}

		static bool cudaSafeCallNoThrow(CUresult err, const char* file, const int line)
		{
			if (!evalErrorNoThrow(err, file, line)) return false;
#if CKL_ALWAYS_SYNC==1
			//insert a device-sync
			err = cudaDeviceSynchronize();
			if (!evalErrorNoThrow(err, file, line)) return false;
#endif
			return true;
		}

		static void cudaCheckError(const char* file, const int line)
		{
			cudaError err = cudaGetLastError();
			if (!evalError(err, file, line)) return;

#if CKL_ALWAYS_SYNC==1
			// More careful checking. However, this will affect performance.
			err = cudaDeviceSynchronize();
			evalError(err, file, line);
#endif
		}
	};

}
CKL_NAMESPACE_END

/**
 * \brief Tests if the cuda library call wrapped inside the bracets was executed successfully, aka returned cudaSuccess.
 * Throws an cuMat::cuda_error if unsuccessfull
 * \param err the error code
 */
#define CKL_SAFE_CALL( err ) CKL_NAMESPACE ::internal::ErrorHelpers::cudaSafeCall( err, __FILE__, __LINE__ )

 /**
* \brief Tests if the cuda library call wrapped inside the bracets was executed successfully, aka returned cudaSuccess
* Returns false iff unsuccessfull
* \param err the error code
*/
#define CKL_SAFE_CALL_NO_THROW( err ) CKL_NAMESPACE ::internal::ErrorHelpers::cudaSafeCallNoThrow( err, __FILE__, __LINE__ )
/**
 * \brief Issue this after kernel launches to check for errors in the kernel.
 */
#define CKL_CHECK_ERROR()    CKL_NAMESPACE ::internal::ErrorHelpers::cudaCheckError( __FILE__, __LINE__ )

#endif