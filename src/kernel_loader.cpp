#include <ckl/kernel_loader.h>

#include <iostream>
#include <filesystem>
#include <nvrtc.h>
#include <sstream>
#include <fstream>
#include <mutex>
#include <thread>

#include "sha1.h"
#include <ckl/errors.h>

CKL_NAMESPACE_BEGIN

namespace fs = std::filesystem;

static void throwOnNvrtcError(nvrtcResult result, const char* file, const int line)
{
    if (result != NVRTC_SUCCESS) {
        throw ckl::cuda_error("NVRTC error at %s:%d : %s",
            file, line, nvrtcGetErrorString(result));
    }
}
#define NVRTC_SAFE_CALL( err ) throwOnNvrtcError( err, __FILE__, __LINE__ )

namespace
{
    struct ContextRAIID
    {
        bool pushed_;
        ContextRAIID(CUcontext ctx)
            : pushed_(false)
        {
            if (ctx != 0) {
                CKL_SAFE_CALL(cuCtxPushCurrent(ctx));
                pushed_ = true;
            }
        }
        ~ContextRAIID()
        {
            if (pushed_) {
                CUcontext dummy;
                CKL_SAFE_CALL(cuCtxPopCurrent(&dummy));
            }
        }
    };
}

void FilesystemLoader::populate(std::vector<NameAndContent>& files)
{
    for (const auto& p : fs::recursive_directory_iterator(root_))
    {
        if (!p.is_regular_file()) continue;
        if (!hasRegex_ || std::regex_match(p.path().string(), regex_))
        {
            try
            {
                std::ifstream t(p.path());
                std::ostringstream ss;
                ss << t.rdbuf();
                std::string buffer = ss.str();
                auto relPath = p.path().lexically_relative(root_).string();
                std::replace(relPath.begin(), relPath.end(), '\\', '/');
                files.push_back(NameAndContent{ relPath, buffer });
            }
            catch (const std::exception& ex)
            {
                std::cerr << "Unable to read file " << p.path() << ": " << ex.what() << std::endl;
            }
        }
    }
}

void ConcatinatingLoader::populate(std::vector<NameAndContent>& files)
{
    for (const auto& l : children_)
        l->populate(files);
}


detail::KernelStorage::KernelStorage(const std::string& kernelName,
                                     const std::vector<NameAndContent>& includeFiles,
                                     const std::string& source,
                                     const std::vector<std::string>& constantNames,
                                     const std::vector<const char*>& compileArgs,
                                     bool verbose)
        : module(nullptr)
        , humanName(kernelName)
        , function(nullptr)
        , minGridSize(0)
        , bestBlockSize(0)
{
    if (verbose) std::cout << "Compile kernel \"" << kernelName << "\"" << std::endl;

    //create program
    nvrtcProgram prog;

    const auto printSourceCode = [](const std::string& s)
    {
        std::istringstream iss(s);
        int lineIndex = 1;
        for (std::string line; std::getline(iss, line); lineIndex++)
        {
            std::cout << "[" << std::setfill('0') << std::setw(5) << lineIndex <<
                "] " << line << "\n";
        }
        std::cout << std::flush;
    };
    if (verbose)
    {
        printSourceCode(source);
    }

    std::vector<const char*> headerContents(includeFiles.size());
    std::vector<const char*> headerNames(includeFiles.size());
    for (size_t i = 0; i < includeFiles.size(); ++i)
    {
        headerContents[i] = includeFiles[i].content.c_str();
        headerNames[i] = includeFiles[i].filename.c_str();
    }
    const char* const* headers = includeFiles.empty() ? nullptr : headerContents.data();
    const char* const* includeNames = includeFiles.empty() ? nullptr : headerNames.data();
    NVRTC_SAFE_CALL(
        nvrtcCreateProgram(&prog,
            source.c_str(),
            "main.cu",
            includeFiles.size(),
            headers,
            includeNames));

    //add kernel name for resolving the native name
    NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernelName.c_str()));
    //add constant names
    for (const auto& var : constantNames)
    {
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, ("&" + var).c_str()));
    }

    //compile
    nvrtcResult compileResult = nvrtcCompileProgram(prog, compileArgs.size(), compileArgs.data());
    // obtain log
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    std::vector<char> log(logSize);
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log[0]));
    if (verbose) std::cout << log.data();
    if (compileResult != NVRTC_SUCCESS)
    {
        nvrtcDestroyProgram(&prog); //ignore possible errors
        printSourceCode(source);
        if (!verbose) std::cout << log.data();
        std::string msg = std::string("Failed to compile kernel:\n") + log.data();
        throw ckl::cuda_error(msg.c_str());
    }

    //optain PTX
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    this->ptxData.resize(ptxSize);
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, this->ptxData.data()));

#if 0
    //test
    std::string ptxStr(this->ptxData.begin(), this->ptxData.end());
    std::cout << "\nPTX:\n" << ptxStr << "\n" << std::endl;
#endif

    //get machine name
    const char* machineName;
    NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, kernelName.c_str(), &machineName));
    this->machineName = machineName;
    for (const auto& var : constantNames)
    {
        std::string humanName = "&" + var;
        const char* machineName;
        NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, humanName.c_str(), &machineName));
        human2machine.emplace(var, std::string(machineName));
    }

    //delete program
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    loadPTX(verbose);
}

detail::KernelStorage::KernelStorage(std::ifstream& i, bool verbose)
    : module(nullptr)
    , function(nullptr)
    , minGridSize(0)
    , bestBlockSize(0)
{
    size_t humanNameSize, machineNameSize, ptxSize;
    i.read(reinterpret_cast<char*>(&humanNameSize), sizeof(size_t));
    humanName.resize(humanNameSize);
    i.read(humanName.data(), humanNameSize);

    i.read(reinterpret_cast<char*>(&machineNameSize), sizeof(size_t));
    machineName.resize(machineNameSize);
    i.read(machineName.data(), machineNameSize);

    i.read(reinterpret_cast<char*>(&ptxSize), sizeof(size_t));
    ptxData.resize(ptxSize);
    i.read(ptxData.data(), ptxSize);

    size_t human2machineSize, strSize;
    std::string key, value;
    human2machine.clear();
    i.read(reinterpret_cast<char*>(&human2machineSize), sizeof(size_t));
    for (size_t j = 0; j < human2machineSize; ++j)
    {
        i.read(reinterpret_cast<char*>(&strSize), sizeof(size_t));
        key.resize(strSize);
        i.read(key.data(), strSize);

        i.read(reinterpret_cast<char*>(&strSize), sizeof(size_t));
        value.resize(strSize);
        i.read(value.data(), strSize);

        human2machine[key] = value;
    }

    loadPTX(verbose);
}

void detail::KernelStorage::loadPTX(bool verbose)
{
    if (verbose) {
        std::cout << "Load module \"" << this->machineName << "\"" << std::endl;
    }

    //TEST:
    CUcontext ctx;
    CKL_SAFE_CALL(cuCtxGetCurrent(&ctx));
    if (verbose) {
        std::cout << "Current context: " << ctx << std::endl;
        std::cout << "Current thread: " << std::this_thread::get_id() << std::endl;
    }

    //load PTX
    unsigned int infoBufferSize = 1024;
    unsigned int errorBufferSize = 1024;
    unsigned int logVerbose = 1;
    std::vector<CUjit_option> options;
    std::vector<void*> values;
    std::unique_ptr<char[]> errorLog = std::make_unique<char[]>(errorBufferSize);
    char* errorLogData = errorLog.get();
    std::unique_ptr<char[]> infoLog = std::make_unique<char[]>(infoBufferSize);
    char* infoLogData = infoLog.get();
    options.push_back(CU_JIT_ERROR_LOG_BUFFER); //Pointer to a buffer in which to print any log messages that reflect errors
    values.push_back(errorLogData);
    options.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES); //Log buffer size in bytes. Log messages will be capped at this size (including null terminator)
    values.push_back(reinterpret_cast<void*>(errorBufferSize));
    options.push_back(CU_JIT_INFO_LOG_BUFFER);
    values.push_back(infoLogData);
    options.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
    values.push_back(reinterpret_cast<void*>(infoBufferSize));
    options.push_back(CU_JIT_TARGET_FROM_CUCONTEXT); //Determines the target based on the current attached context (default)
    values.push_back(0); //No option value required for CU_JIT_TARGET_FROM_CUCONTEXT
    options.push_back(CU_JIT_LOG_VERBOSE);
    values.push_back(reinterpret_cast<void*>(logVerbose));
    auto err = cuModuleLoadDataEx(&this->module, this->ptxData.data(), options.size(), options.data(), values.data());
    if (infoLogData[0] && verbose)
    {
        std::cout << infoLog.get() << std::endl;
    }
    if (errorLog[0]) {
        std::cerr << "Compiler error: " << errorLog.get() << std::endl;
    }
    CKL_SAFE_CALL(err);

    //get cuda function and constants
    CKL_SAFE_CALL(cuModuleGetFunction(&this->function, this->module, this->machineName.data()));
    for (const auto& e : human2machine)
    {
        if (verbose) {
            std::cout << "Fetch address for constant variable \"" << e.first
                << "\", machine name \"" << e.second << "\"" << std::endl;
        }
        CUdeviceptr addr;
        CKL_SAFE_CALL(cuModuleGetGlobal(&addr, nullptr, module, e.second.data()));
        constants[e.first] = addr;
        if (verbose)
            std::cout << "constant variable " << e.first << " has device pointer 0x"
            << std::hex << addr << std::dec << std::endl;
    }

    CKL_SAFE_CALL(cuOccupancyMaxPotentialBlockSize(
        &minGridSize, &bestBlockSize, function, NULL, 0, 0));

    if (verbose) {
        std::cout << "Module \"" << this->machineName << "\" loaded successfully"
            << ", block size: " << bestBlockSize << std::endl;
    }
}

detail::KernelStorage::~KernelStorage()
{
    CUresult err = cuModuleUnload(this->module);
    if (err == CUDA_ERROR_DEINITIALIZED)
    {
        //ignore silently, we are terminating the program
    }
    else if (err != CUDA_SUCCESS) {
        const char* pStr;
        cuGetErrorString(err, &pStr);
        const char* pName;
        cuGetErrorName(err, &pName);
        std::stringstream ss;
        std::cerr << "Cuda error " << pName << " when unloading module for kernel " << machineName << ":" << pStr << std::endl;
    }
}

void detail::KernelStorage::save(std::ofstream& o) const
{
    size_t humanNameSize = humanName.size();
    o.write(reinterpret_cast<const char*>(&humanNameSize), sizeof(size_t));
    o.write(humanName.c_str(), humanNameSize);

    size_t machineNameSize = machineName.size();
    o.write(reinterpret_cast<const char*>(&machineNameSize), sizeof(size_t));
    o.write(machineName.c_str(), machineNameSize);

    size_t ptxSize = ptxData.size();
    o.write(reinterpret_cast<const char*>(&ptxSize), sizeof(size_t));
    o.write(ptxData.data(), ptxSize);

    size_t human2machineSize = human2machine.size();
    o.write(reinterpret_cast<const char*>(&human2machineSize), sizeof(size_t));
    for (const auto& e : human2machine)
    {
        size_t strSize = e.first.size();
        o.write(reinterpret_cast<const char*>(&strSize), sizeof(size_t));
        o.write(e.first.data(), strSize);

        strSize = e.second.size();
        o.write(reinterpret_cast<const char*>(&strSize), sizeof(size_t));
        o.write(e.second.data(), strSize);
    }
}


CUfunction KernelFunction::fun() const
{
    return storage_->function;
}

int KernelFunction::minGridSize() const
{
    return storage_->minGridSize;
}

int KernelFunction::bestBlockSize() const
{
    return storage_->bestBlockSize;
}

CUdeviceptr KernelFunction::constant(const std::string& name) const
{
    const auto it = storage_->constants.find(name);
    if (it == storage_->constants.end())
        return 0;
    else
        return it->second;
}

std::string KernelFunction::name() const
{
    return storage_->humanName;
}

void KernelFunction::fillConstantMemory(const std::string& name, const void* dataHost, size_t size, bool async,
                                        CUstream stream)
{
    CUdeviceptr ptr = constant(name);
    if (ptr == 0)
        throw cuda_error("Constant variable with name '%s' not found", name);

    if (async) 
    {
        CKL_SAFE_CALL(cuMemcpyHtoDAsync(ptr, dataHost, size, stream));
    }
    else
    {
        CKL_SAFE_CALL(cuMemcpyHtoD(ptr, dataHost, size));
    }
}

void KernelFunction::call(unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ, unsigned blockDimX,
    unsigned blockDimY, unsigned blockDimZ, unsigned sharedMemBytes, CUstream hStream, void** kernelParams)
{
    CUcontext ctx;
    CKL_SAFE_CALL(cuCtxGetCurrent(&ctx));
    ContextRAIID contextRaiid(ctx != ctx_ ? ctx_ : nullptr);

    CKL_SAFE_CALL(cuLaunchKernel(fun(), gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, nullptr));
}


KernelLoader::KernelLoader()
{
    //query compute capability
    cudaDeviceProp props {0};
    CKL_SAFE_CALL(cudaGetDeviceProperties(&props, 0));
    computeMajor_ = props.major;
    computeMinor_ = props.minor;
    std::cout << "Compiling kernels for device '" << props.name << "' with compute capability " <<
        props.major << "." << props.minor << std::endl;
    computeArchitecture_ = internal::Format::format("--gpu-architecture=compute_%d%d", computeMajor_, computeMinor_);

#ifdef CKL_NVCC_INCLUDE_DIR
    std::cout << "NVCC include directory: " << CKL_STR(CKL_NVCC_INCLUDE_DIR) << std::endl;
#else
    std::cout << "Warning: no NVCC include directory specified. Compiling kernels will likely fail" << std::endl;
#endif

    compileOptions_ = {
        computeArchitecture_.c_str(),
        "-std=c++17",
        "--use_fast_math",
        "--generate-line-info",
        "-Xptxas",
        "-v",
#ifdef CKL_NVCC_INCLUDE_DIR
        "-I", CKL_STR(CKL_NVCC_INCLUDE_DIR),
#endif
        "-D__NVCC__=1",
        "-DCUDA_NO_HOST=1",
    };

    //query context
    CKL_SAFE_CALL(cuCtxGetCurrent(&ctx_));
    std::cout << "Current CUDA Driver context: " << ctx_ << std::endl;
}

KernelLoader::~KernelLoader()
{
    cleanup();
}

KernelLoader& KernelLoader::Instance()
{
    static KernelLoader INSTANCE;
    return INSTANCE;
}

void KernelLoader::setFileLoader(const std::shared_ptr<IFileLoader>& loader)
{
    fileLoader_ = loader;
    reloadCudaKernels();
}

std::optional<std::string> KernelLoader::findFile(const std::string& filename)
{
    for (const auto& nc : includeFiles_)
    {
        if (nc.filename == filename)
        {
            return nc.content;
        }
    }
    return {};
}

void KernelLoader::setCacheDir(const std::filesystem::path& path)
{
    cacheDirectory_ = path;
    reloadCudaKernels();
    loadKernelCache(false);
}

void KernelLoader::disableCudaCache()
{
    cacheDirectory_ = "";
}

void KernelLoader::reloadCudaKernels()
{
    includeFiles_.clear();
    if (loadCUDASources())
    {
        //files changed, clear kernels
        kernelStorage_.clear();
    }
}

void KernelLoader::cleanup()
{
    includeFiles_.clear();
    kernelStorage_.clear();
}

bool KernelLoader::loadCUDASources()
{
    if (!includeFiles_.empty()) {
        //already loaded
        return false;
    }

    // load files
    if (fileLoader_)
    {
        fileLoader_->populate(includeFiles_);
    }

    // compute hashes
    SHA1 sha1;
    for (const auto& e : includeFiles_)
        sha1.update(e.content);

    std::string previousHash = includeFilesHash_;
    includeFilesHash_ = sha1.final();
    return previousHash != includeFilesHash_;
}

std::optional<KernelFunction> KernelLoader::getKernel(
    const std::string& kernelName, const std::string& sourceCode,
    const std::vector<std::string>& constantNames, int flags)
{
    bool verbose = flags & CompileVerboseLogging;

    //check if we are in a multi-threaded environment
    CUcontext ctx;
    CKL_SAFE_CALL(cuCtxGetCurrent(&ctx));
    if (ctx_ == nullptr)
    {
        //no context found during initialization, set this one as the default one
        ctx_ = ctx;
    }
    if (ctx != ctx_ && verbose)
    {
        std::cout << "Attempt to call getKernel() from a different thread or different context than where the KernelLoader was created from. " <<
            "Expected context: " << ctx_ << ", current context: " << ctx << ". " <<
            "Restore the old context now." << std::endl;
    }
    ContextRAIID contextRaiid(ctx != ctx_ ? ctx_ : nullptr);

    loadKernelCache(verbose);

    SHA1 sha;
    sha.update(kernelName);
    sha.update(sourceCode);
    sha.update(std::to_string(flags));
    const std::string kernelKey = sha.final();

    const auto it = kernelStorage_.find(kernelKey);
    if (it == kernelStorage_.end())
    {
        //kernel not found in the cache, recompile it

        //assemble compile options
        std::vector<const char*> opts = compileOptions_;
        if (flags & CompileDebugMode)
        {
            opts.push_back("-G");
        }

        //compile
        try {
            const auto storage = std::make_shared<detail::KernelStorage>(
                kernelName, includeFiles_, sourceCode, 
                constantNames, opts, verbose);

            kernelStorage_.emplace(kernelKey, storage);
            saveKernelCache();
            return KernelFunction(storage, ctx_);
        }
        catch (std::exception& ex)
        {
            if (flags & CompileThrowOnError) {
                throw;
            }
            else {
                std::cerr << "Unable to compile kernel: " << ex.what() << std::endl;
                reloadCudaKernels(); //so that in the next iteration, we can compile again
                return {};
            }
        }
    }
    else
    {
        //kernel found, return immediately
        return KernelFunction(it->second, ctx_);
    }
}

std::string KernelLoader::MainFile(const std::string& filename)
{
    return "#include \"" + filename + "\"\n";
}


void KernelLoader::saveKernelCache()
{
    if (cacheDirectory_.empty()) return;

    if (!exists(cacheDirectory_)) {
        if (!create_directory(cacheDirectory_)) {
            std::cerr << "Unable to create cache directory " << absolute(cacheDirectory_) << std::endl;
            return;
        }
        else
            std::cout << "Cache directory created at " << absolute(cacheDirectory_) << std::endl;
    }

    fs::path cacheFile = cacheDirectory_ / (includeFilesHash_ + ".kernel");
    std::ofstream o(cacheFile, std::ofstream::binary);
    if (!o.is_open())
    {
        std::cerr << "Unable to open cache file " << absolute(cacheFile) << " for writing" << std::endl;
        return;
    }
    o.write(reinterpret_cast<const char*>(&KERNEL_CACHE_MAGIC), sizeof(int));
    size_t entrySize = kernelStorage_.size();
    o.write(reinterpret_cast<const char*>(&entrySize), sizeof(size_t));
    for (const auto e : kernelStorage_)
    {
        size_t kernelNameSize = e.first.size();
        o.write(reinterpret_cast<const char*>(&kernelNameSize), sizeof(size_t));
        o.write(e.first.data(), kernelNameSize);
        e.second->save(o);
    }
    std::cout << entrySize << " kernels written to the cache file " << cacheFile << std::endl;
}

void KernelLoader::loadKernelCache(bool verbose)
{
    if (!kernelStorage_.empty()) return; //already loaded

    //load cuda source files and updates the SHA1 hash
    loadCUDASources();

    if (cacheDirectory_.empty())
    {
        //no cache directory specified.
        return;
    }

    fs::path cacheFile = cacheDirectory_ / (includeFilesHash_ + ".kernel");
   
    if (exists(cacheFile))
    {
        std::cout << "Read from cache " << cacheFile << std::endl;
        std::ifstream i(cacheFile, std::ifstream::binary);
        if (!i.is_open())
        {
            std::cerr << "Unable to open file" << std::endl;
            return;
        }
        unsigned int magic;
        i.read(reinterpret_cast<char*>(&magic), sizeof(int));
        if (magic != KERNEL_CACHE_MAGIC)
        {
            std::cerr << "Invalid magic number, wrong file type or file is corrupted" << std::endl;
            return;
        }
        size_t entrySize;
        i.read(reinterpret_cast<char*>(&entrySize), sizeof(size_t));
        for (size_t j = 0; j < entrySize; ++j)
        {
            size_t kernelNameSize;
            std::string kernelName;
            i.read(reinterpret_cast<char*>(&kernelNameSize), sizeof(size_t));
            kernelName.resize(kernelNameSize);
            i.read(kernelName.data(), kernelNameSize);

            const auto storage = std::make_shared<detail::KernelStorage>(i, verbose);
            kernelStorage_.emplace(kernelName, storage);
        }
        std::cout << entrySize << " kernels loaded from cache" << std::endl;
    }
}



CKL_NAMESPACE_END
