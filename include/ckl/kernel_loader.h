#pragma once

#include <cuda.h>
#include <filesystem>
#include <map>
#include <fstream>
#include <functional>
#include <optional>
#include <vector>
#include <regex>

#include "internal_common.h"

CKL_NAMESPACE_BEGIN

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

class FilesystemLoader : public IFileLoader
{
    const std::filesystem::path root_;
    const bool hasRegex_;
    const std::regex regex_;

public:
    virtual ~FilesystemLoader() = default;

    explicit FilesystemLoader(const std::filesystem::path& root)
        : root_(root), hasRegex_(false)
    {}
    FilesystemLoader(const std::filesystem::path& root, const std::string& regex)
        : root_(root), hasRegex_(true), regex_(regex)
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

        CUmodule module;
        std::vector<char> ptxData;
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
            bool verbose);

        //loads the pre-compiled kernel from the cache file
        KernelStorage(std::ifstream& i, bool verbose);

        void loadPTX(bool verbose);

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
    friend class KernelLoader;
    explicit KernelFunction(const std::shared_ptr<detail::KernelStorage>& storage)
        : storage_(storage) {}
public:
    KernelFunction() = default;
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
     */
    KernelLoader();

    ~KernelLoader();

    /**
     * The global singleton instance
     */
    static KernelLoader& Instance();

    /**
     * Sets the file loader used to load the available include files.
     * This automatically calls reloadCudaKernels().
     */
    void setFileLoader(const std::shared_ptr<IFileLoader>& loader);

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
        /**
         * Activates verbose logging.
         */
        CompileVerboseLogging = 4
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
        const std::vector<std::string>& constantNames,
        int flags = CompileThrowOnError);

private:

    //Loads the CUDA source file
    //Returns true if the source files have changed
    bool loadCUDASources();

    void saveKernelCache();
    void loadKernelCache(bool verbose);
    static constexpr unsigned int KERNEL_CACHE_MAGIC = 0x61437543u; //CuCa

    std::filesystem::path cacheDirectory_;

    std::shared_ptr<IFileLoader> fileLoader_;
    std::vector<NameAndContent> includeFiles_;
    std::string includeFilesHash_;

    std::map<std::string, std::shared_ptr<detail::KernelStorage>> kernelStorage_;
};
typedef std::shared_ptr<KernelLoader> KernelLoader_ptr;

CKL_NAMESPACE_END
