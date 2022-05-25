# CUDA Kernel Loader

Tired of large switch statements to compile from runtime arguments to static arguments for template instancing?
Tired of long compilation times?
Tired of being limited to the options included during compilation?

Fret no more, with this library, you can configure, assemble and compile CUDA kernels during runtime.



### Example:
Assume you have a kernel with a lot of compile-time switches via template in a file `my_kernel.cu`

```c++
//my_kernel.cu
template<bool Switch1, int Variable2, MyEnum Enum3>
__global__ MyKernel(int arg1, float* arg2) {...}
```

Traditionally, you would need to instantiate all possible combinations during compile time and then use nested switch-statements to translate the arguments.

```c++
//traditional_launch.cpp
void launchKernel(bool Switch1, int Variable2, MyEnum Enum3, int arg1, float* arg2)
{
    unsigned gridDim = 1;
    unsigned blockDim = 1;
    switch (Enum3)
    {
    case MyEnum::Value1:
        switch (Variable2):
        {
        case 1:
            if (Swtich1)
                //finally the kernel
                MyKernel<true, 1, MyEnum::Value1><<<gridDim,blockDim>>>(arg1, arg2);
            else
                MyKernel<false, 1, MyEnum::Value1><<<gridDim,blockDim>>>(arg1, arg2);
            break;
        case 2:
            //more lines for all possible values of 'Variable2'
        default:
            throw exception("This value is outside of what was specified during compilation :( ");
        }
        break;
    case MyEnum::Value2:
        //more lines for all possible values of 'Enum3'
        //need to duplicate all of above, but change MyEnum.
    }
}
```

This is a lot of code duplication and code bloat. It also restricts the possible values of something like `Variable2` to values specified during compilation.
With this library, you can simplify this to:

```c++
//runtime_compilation.cpp
#include <ckl/kernel_loader.h>
void launchKernel(bool Switch1, int Variable2, MyEnum Enum3, int arg1, float* arg2)
{
    ckl::KernelLoader kl = ... //see below
    
    //generate kernel name, e.g. MyKernel<true, 1, MyEnum::Value1>
    std::string kernelName = "MyKernel<" + to_string(Switch1) + ", " + to_string(Variable2) + ", " + to_string(Enum3) + ">";
    //compile the kernel
    std::string kernelCode = ckl::KernelLoader::MainFile("my_kernel.cu");
    auto fun = kl->getKernel(kernelName, kernelCode);
    
    //launch kernel
    unsigned gridDim = 1;
    unsigned blockDim = 1;
    fun->call(gridDim, blockDim, 0, nullptr, 
       /* now follows the arguments to the kernel*/
       arg1, arg2);
}
```



The code behind this repository was used in the following projects:

- Fast Neural Representations for Direct Volume Rendering (https://github.com/shamanDevel/fV-SRN)
- Differentiable Direct Volume Rendering (https://github.com/shamanDevel/DiffDVR)

For future projects and less code duplications, I pulled the code out and into this library now.



## Requirements

 - CMake 3.4 or higher

 - CUDA SDK 10.0 or higher
   Note: the SDK must be installed, not just the runtime. Otherwise, NVRTC can't be found

 - A C++17 compatible compiler


## Installation

- Clone this repository or add it as a submodule
- Use `add_subdirectory` in CMake to register this project
- Link against the target `ckl` and enjoy. This sets up the include directory and links against the library
  CMake `target_link_libraries(your-library PRIVATE ckl)`

CMake Options:

- CKL_SHARED: if set to ON, build the library as a shared library (.so / .dll) instead of a static library
- CKL_ARCH_FLAGS: Available if the CUDA architecture couldn't be automatically specified. Then set the compute architecture here. Examples: `"75"` for RTX 20xx cards, `"61"` for GTX 10xx cards



## Project structure of `ckl`

The main entry point is the file `ckl/kernel_loader.h`

`class ckl::KernelLoader`:
Describes how the kernels are compiled and what include files are available. The compiled kernels are cached, optionally on hard-disk, to speed-up subsequent program executions.
- `KernelLoader()`:
  Constructs a new kernel loader. Usually, only one instance is used per project and shared among all usages.
- `setFileLoader(std::shared_ptr<IFileLoader>),  reloadCudaKernels()`
  Sets the file loader that specifies, what include files are available in the kernels. See below for more details. With `reloadCudaKernels()`, the include files are scanned again and kernels will be recompiled upon next usage if the source code has changed.
- `setCacheDir(std::filesystem::path)`
  Sets the folder for caching compiled kernels on disk. This way, kernels don't need to be recompiled during the next run. If you don't call this method or disable the cache with `disableCudaCache()`, kernels are still cached in-memory for the runtime of the current program, but not saved persistently to disk
- `optional<KernelFunction> getKernel(string kernelName, string sourceCode, ...)`
  The main entry point for compiling and retrieving CUDA kernels. For more details, see the section on Applications.
  

`class ckl::KernelFunction`:
Holds a single compiled CUDA kernel.
Best practice: don't store this instance, instead fetch it again on every usage with `KernelLoader::getKernel`. This allows to react to file changes on the inputs. This saved me a lot of time with debugging the CUDA kernels. I could simply modify the source code, call `KernelLoader::reloadCudaKernels` (a button in my UI) and the next kernel calls used the new source code. No need to restart the program and set-up the inputs again.
- `bestBlockSize()`:
  The best block size to fully utilize the GPU. Larger block size might run out of available registers.
- `fillConstantMemory**(string name, T data)`
  Sets constant memory in the CUDA kernel. This is an alternative to passing inputs via kernel function arguments.
- `call(unsigned gridDim, unsigned blockDim, unsigned sharedMemBytes, CUstream hStream, Args... args)`:
  Launches the CUDA kernel with the given grid specification, shared memory and CUDA stream. The arguments the the kernel are passed via variadic template arguments.

`class ckl::IFileLoader`:
Abstract class that specifies, how the include files for the kernels are found.
Available implementations, see also `tests/test_loaders.cpp`:

- `ckl::FilesystemLoader`: loads all files recursivly from a given root folder. Optionally filters the files via a regex
- `ckl::CMRCLoader`: loads the files in the embedded filesystem obtained by the CMRC resource compiler (https://github.com/vector-of-bool/cmrc).
  This is usefull for deployment as the kernel files are now all embedded in the executable and you don't need to carry around the single plain-text source files. However, reloading the source files upon modification is not possible here.
- `ConcatinatingLoader`: concatenates multiple file loaders. Typical usage is to include multiple FilesystemLoaders if you have multiple root directories with kernel files.

## Applications

TODO: demonstrate code generation + template specification

Note: the string conversions have to be implemented by the user.
For enums, see e.g. enum_name by https://github.com/Neargye/magic_enum

