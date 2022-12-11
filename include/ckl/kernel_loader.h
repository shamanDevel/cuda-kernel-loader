#pragma once

#include <cuda.h>
#include <filesystem>
#include <map>
#include <optional>
#include <vector>
#include <regex>
#include <memory>
#include <spdlog/spdlog.h>

#include "common.h"

CKL_NAMESPACE_BEGIN

typedef std::shared_ptr<spdlog::logger> logger_t;

/**
 * Holder for the filename, path separation via '/',
 * and the contents of the file.
 * This is passed to the compiler as includeable files.
 */
struct NameAndContent
{
    std::string filename;
    std::string content;
};

/**
 * Base class for file loaders.
 * Its purpose is to populate the list of NameAndContent-instances
 * that are available during compilation of kernels.
 */
class IFileLoader
{
public:
    virtual ~IFileLoader() = default;

    /**
     * Populates the list of files that are available during compilation.
     */
    virtual void populate(std::vector<NameAndContent>& files) = 0;
};

/**
 * FileLoader implementation that reads all files recursively
 * starting from a root path.
 * Optionally, the files that are included can be filtered using a regex expression.
 */
class FilesystemLoader : public IFileLoader
{
    const std::filesystem::path root_;
    const bool hasRegex_;
    const std::regex regex_;

public:
    explicit FilesystemLoader(std::filesystem::path root)
        : root_(std::move(root)), hasRegex_(false)
    {}
    FilesystemLoader(std::filesystem::path root, const std::string& regex)
        : root_(std::move(root)), hasRegex_(true), regex_(regex)
    {}

    void populate(std::vector<NameAndContent>& files) override;
};

/**
 * Concatenates multiple file loaders.
 * This way, multiple source roots (if using FilesystemLoader) can be used
 * together.
 */
class ConcatinatingLoader : public IFileLoader
{
    const std::vector<std::shared_ptr<IFileLoader>> children_;

public:
    explicit ConcatinatingLoader(std::vector<std::shared_ptr<IFileLoader>> children)
        : children_(std::move(children))
    {}

    void populate(std::vector<NameAndContent>& files) override;
};

namespace detail
{
    /**
     * The backend storage of the kernel
     */
    struct KernelStorage
    {
        typedef std::map<std::string, CUdeviceptr> constants_t;

        const logger_t logger;
        CUmodule module;
        std::vector<char> ptxData;
        std::string humanName;
        std::string machineName;
        CUfunction function;
        int minGridSize;
        int bestBlockSize;
        constants_t constants;
        std::map<std::string, std::string> human2machine;

        //creates and compiles the kernel
        KernelStorage(
            const std::string& kernelName,
            const std::vector<NameAndContent>& includeFiles,
            const std::string& source,
            const std::vector<std::string>& constantNames,
            const std::vector<const char*>& compileArgs,
            logger_t logger);

        //loads the pre-compiled kernel from the cache file
        KernelStorage(std::ifstream& i, logger_t logger);

        void loadPTX();

        //unloads the kernel
        ~KernelStorage();

        //saves the kernel to the cache file
        void save(std::ofstream& o) const;
    };
}

class KernelLoader;

/**
 * Stores the compiled kernel and descriptors on how to launch it.
 * Note: you should not store this function for a longer time, but rather
 * fetch it with KernelLoader::getKernel() again.
 * Otherwise, it won't get updated when the source files are changed.
 */
class KernelFunction
{
    std::shared_ptr<detail::KernelStorage> storage_;
    CUcontext ctx_ = nullptr;
    friend class KernelLoader;
    KernelFunction(const std::shared_ptr<detail::KernelStorage>& storage, CUcontext ctx)
        : storage_(storage), ctx_(ctx) {}
public:
    KernelFunction() = default;
    /**
     * \brief Returns true iff the kernel function is defined (compiled and loaded).
     * False in the case of created by the default constructor.
     */
    [[nodiscard]] bool defined() const { return storage_ != nullptr; }
    /**
     * \brief Returns the native CUDA function handle
     */
    [[nodiscard]] CUfunction fun() const;
    /**
     * The minimal grid size to fully utilize the GPU
     */
    [[nodiscard]] int minGridSize() const;
    /**
     * The best block size to fully utilize the GPU.
     * Larger block size might run out of available registers.
     */
    [[nodiscard]] int bestBlockSize() const;
    /**
     * \brief Returns the device pointer for the constant variable with the given name
     * or '0' if not found.
     */
    [[nodiscard]] CUdeviceptr constant(const std::string& name) const;
    /**
     * \brief Returns the human name of this kernel
     */
    [[nodiscard]] std::string name() const;

private:
    void fillConstantMemory(const std::string& name, const void* dataHost, size_t size, bool async, CUstream stream);

public:
    /**
     * Fills the constant memory with name 'name'
     * with the data provided in 'data' on the host/cpu.
     *
     * Effectively calls KernelFunction::constant() to fetch the address
     * of the constant variable and then calls cuMemcpyHtoD to copy
     * the host memory to the device memory.
     *
     * If an error occurs (constnat variable not found or illegal memcpy, an exception is thrown)
     *
     * \param name the name of the constant variable
     * \param data the instance to copy into the constant variable
     */
    template<typename T>
    void fillConstantMemorySync(const std::string& name, const T& data)
    {
        fillConstantMemory(name, &data, sizeof(T), false, nullptr);
    }

    /**
     * Fills the constant memory with name 'name'
     * with an array of variables provided in 'data' on the host/cpu.
     *
     * Effectively calls KernelFunction::constant() to fetch the address
     * of the constant variable and then calls cuMemcpyHtoD to copy
     * the host memory to the device memory.
     *
     * If an error occurs (constnat variable not found or illegal memcpy, an exception is thrown)
     *
     * \param name the name of the constant variable
     * \param dataArray the pointer to first entry in the array to copy
     * \param size the number of entries in the array
     */
    template<typename T>
    void fillConstantMemorySync(const std::string& name, const T* dataArray, size_t size)
    {
        fillConstantMemory(name, dataArray, sizeof(T)*size, false, nullptr);
    }

    /**
     * Fills the constant memory with name 'name'
     * with the data provided in 'data' on the host/cpu.
     * This is the async version of KernelFunction::fillConstantMemorySync!
     *
     * Effectively calls KernelFunction::constant() to fetch the address
     * of the constant variable and then calls cuMemcpyHtoD to copy
     * the host memory to the device memory.
     *
     * If an error occurs (constnat variable not found or illegal memcpy, an exception is thrown)
     *
     * \param name the name of the constant variable
     * \param data the instance to copy into the constant variable
     * \param stream the cuda stream for tracking the asynchronous memcopy
     */
    template<typename T>
    void fillConstantMemoryAsync(const std::string& name, const T& data, CUstream stream)
    {
        fillConstantMemory(name, &data, sizeof(T), true, stream);
    }

    /**
     * Fills the constant memory with name 'name'
     * with an array of variables provided in 'data' on the host/cpu.
     * This is the async version of KernelFunction::fillConstantMemorySync!
     *
     * Effectively calls KernelFunction::constant() to fetch the address
     * of the constant variable and then calls cuMemcpyHtoD to copy
     * the host memory to the device memory.
     *
     * If an error occurs (constnat variable not found or illegal memcpy, an exception is thrown)
     *
     * \param name the name of the constant variable
     * \param dataArray the pointer to first entry in the array to copy
     * \param size the number of entries in the array
     * \param stream the cuda stream for tracking the asynchronous memcopy
     */
    template<typename T>
    void fillConstantMemoryAsync(const std::string& name, const T* dataArray, size_t size, CUstream stream)
    {
        fillConstantMemory(name, dataArray, sizeof(T) * size, true, stream);
    }

    void callRaw(unsigned int gridDimX,
        unsigned int gridDimY,
        unsigned int gridDimZ,
        unsigned int blockDimX,
        unsigned int blockDimY,
        unsigned int blockDimZ,
        unsigned int sharedMemBytes,
        CUstream hStream,
        void** kernelParams);

    /**
     * Launches the kernel with a 1D grid
     * of 'gridDim' blocks with 'blockDim' threads per block.
     * Per block, 'sharedMemBytes' of shared memory is dynamically specified (often zero),
     * and the kernel is added to the stream 'hStream'.
     * The arguments to the kernel are specified in the variadic template parameter.
     *
     * \param gridDim the grid dimension, i.e. number of blocks
     * \param blockDim the block dimension, i.e. number of threads per block.
     *    See also KernelFunction::bestBlockSize()
     * \param sharedMemBytes the amount of dynamic shared memory (often zero)
     * \param hStream the stream to enqueue this kernel invocation
     * \param args the arguments passed to the kernel
     */
    template<typename... Args>
    void call(
        unsigned int gridDim, unsigned int blockDim, 
        unsigned int sharedMemBytes, CUstream hStream,
        Args... args)
    {
        //fetch addresses of the arguments
        void* argv[] = { std::addressof(args)... };

        // launch
        callRaw(gridDim, 1, 1, blockDim, 1, 1, sharedMemBytes, hStream, argv);
    }
};

/**
 * \brief The central kernel loader to compile kernels.
 *
 * You can construct custom instances, e.g. if you have multiple sets of
 * file loaders, or use the global instance.
 * Instances of this class can't be moved or copied. For custom instances,
 * use KernelLoader_ptr instead.
 */
class KernelLoader 
{
public:
    typedef detail::KernelStorage::constants_t constants_t;

    KernelLoader(const KernelLoader& other) = delete;
    KernelLoader(KernelLoader&& other) noexcept = delete;
    KernelLoader& operator=(const KernelLoader& other) = delete;
    KernelLoader& operator=(KernelLoader&& other) noexcept = delete;
    
public:
    /**
     * Constructs a new kernel loader.
     * In usual use cases, the global instance KernelLoader::Instance()
     * is probably the better choice to have a unified caching and file loading system.
     *
     * \see KernelLoader::Instance()
     * \param logger the logger to use. If NULL, a console logger is used
     */
    explicit KernelLoader(logger_t logger = nullptr);

    ~KernelLoader();

    /**
     * The global singleton instance
     */
    static KernelLoader& Instance();

    /**
     * Returns the logger instance used to report compile logs (debug) or errors
     */
    [[nodiscard]] logger_t getLogger() const;

    /**
     * Sets the log level.
     * The kernel loader uses the following levels:
     *  - debug: verbose info on the kernel names and source code
     *  - info: a new kernel is compiled
     *  - error: compilation errors
     */
    void setLogLevel(spdlog::level::level_enum level);

    /**
     * Sets the file loader used to load the available include files.
     * This automatically calls reloadCudaKernels().
     */
    void setFileLoader(const std::shared_ptr<IFileLoader>& loader);

    /**
     * Returns the content of the given file from the registered file loader,
     * if found.
     */
    std::optional<std::string> findFile(const std::string& filename);

    /**
     * Sets the cache directory and enables caching.
     * Kernels are stored persistently there to reduce the re-compilation
     * times between program runs.
     */
    void setCacheDir(const std::filesystem::path& path);

    /**
     * The default directly name for the kernel cache.
     * If multiple libraries use different KernelLoader instances,
     * they can use this default to share the same cache directory at least.
     * Filenames are unique, different instances with different source files
     * won't collide (unless you run into a SHA1-collision).
     */
    static constexpr const char* DEFAULT_CACHE_DIR = "kernel_cache";

    /**
     * Disables the cache file.
     */
    void disableCudaCache();

    /**
     * \brief Reloads the source files from the file locator.
     * If files have changed, clear the cached kernels.
     */
    void reloadCudaKernels();

    /**
     * \brief Manual cleanup, unloads all CUDA modules.
     * If not explicitly called, called in the destructor.
     * Can also be used to force-reload all files and force-recompile all kernels.
     */
    void cleanup();

    /**
     * bit-flags for getKernel()
     */
    enum CompilationFlags
    {
        /**
         * Enables debug-mode compilation.
         * Disables optimizations and allows to use CUDA debuggers
         */
        CompileDebugMode = 1,
        /**
         * Instead of returning an empty optional if the compilation fails,
         * throw an std::runtime_error instead.
         */
        CompileThrowOnError = 2,
    };

    /**
     * \brief Retrieves the CUfunction for the specified kernel.
     * The kernel is compiled if needed.
     *
     * Note: the kernel name must be unique as it is used to cache
     * the compiled kernel in the kernel storage.
     * 
     * \param kernelName the name of the kernel as you would declare it in code,
     *   i.e. the name of the __global__ function, including template arguments.
     * \param sourceCode the source code of the kernel.
     *   This is useful if you want to include code generation.
     *	 If you just want to load an include file from the file loader that
     *	 already contains the kernel (i.e. a __global__ function), then
     *	 pass the result of KernelLoader::MainFile(const std::string&) here.
     * \param constantNames list of constant variables that are converted to device names
     *   and accessible via \ref KernelFunction::constant(std::string).
     * \param flags a combination of CompilationFlags specifying options during compilation.
     *   Default: CompileThrowOnError. The returned optional is never empty,
     *	  an exception is thrown if the compilation fails.
     * \return the CUfunction or empty if not found / unable to compile
     */
    [[nodiscard]] std::optional<KernelFunction> getKernel(
        const std::string& kernelName,
        const std::string& sourceCode,
        const std::vector<std::string>& constantNames = {},
        int flags = CompileThrowOnError);

    /**
     * Constructs the source code to be passed as 'sourceCode' argument to
     * KernelLoader::getKernel, if just the specified file
     * should be used as the main file.
     * Note that this file must be available in the file locators.
     *
     * Effectively returns
     * <code>#include "{filename}"</code>
     */
    static std::string MainFile(const std::string& filename);

    /**
     * Returns the major revision number defining the device's compute capability
     */
    [[nodiscard]] int computeMajor() const { return computeMajor_; }

    /**
     * Returns the major revision number defining the device's compute capability
     */
    [[nodiscard]] int computeMinor() const { return computeMinor_; }

    /**
     * Returns the compute capability of the current device.
     * Computed as 10*major+minor. Examples:
     *  - 61 for Pascal architecture
     *  - 75 for the Turing architecture
     *  - 86 for Ampere
     */
    [[nodiscard]] int computeCapability() const { return computeMajor_; }

private:

    //Loads the CUDA source file
    //Returns true if the source files have changed
    bool loadCUDASources();

    void saveKernelCache();
    void loadKernelCache();
    static constexpr unsigned int KERNEL_CACHE_MAGIC = 0x61437543u; // CuCa

    CUcontext ctx_;
    int computeMajor_;
    int computeMinor_;
    std::string computeArchitecture_;
    std::vector<const char*> compileOptions_;
    logger_t logger_;

    std::filesystem::path cacheDirectory_;

    std::shared_ptr<IFileLoader> fileLoader_;
    std::vector<NameAndContent> includeFiles_;
    std::string includeFilesHash_;

    std::map<std::string, std::shared_ptr<detail::KernelStorage>> kernelStorage_;
};
typedef std::shared_ptr<KernelLoader> KernelLoader_ptr;

CKL_NAMESPACE_END
