#include <ckl/kernel_loader.h>

#include <filesystem>
#include <nvrtc.h>
#include <sstream>
#include <fstream>
#include <mutex>
#include <thread>
#include <spdlog/spdlog.h>
#include "spdlog/sinks/stdout_color_sinks.h"

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
                                     logger_t logger)
        : logger(logger)
        , module(nullptr)
        , humanName(kernelName)
        , function(nullptr)
        , minGridSize(0)
        , bestBlockSize(0)
{
    logger->info("Compile kernel {}", kernelName);

    //create program
    nvrtcProgram prog;

    const auto printSourceCode = [](const std::string& s)
    {
        std::stringstream ss;
        std::istringstream iss(s);
        int lineIndex = 1;
        for (std::string line; std::getline(iss, line); lineIndex++)
        {
            ss << "[" << std::setfill('0') << std::setw(5) << lineIndex <<
                "] " << line << "\n";
        }
        return ss.str();
    };
    if (logger->should_log(spdlog::level::debug)) {
        logger->debug("Source code: {}", printSourceCode(source));
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
    logger->debug("Compilation log: {}", log.data());
    if (compileResult != NVRTC_SUCCESS)
    {
        nvrtcDestroyProgram(&prog); //ignore possible errors
        printSourceCode(source);
        if (!logger->should_log(spdlog::level::debug))
        {
            logger->error("Compilation log: {}", log.data());
        }
        std::string msg = std::string("Failed to compile kernel:\n") + log.data();
        throw ckl::cuda_error(msg.c_str());
    }

    //optain PTX
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    this->ptxData.resize(ptxSize);
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, this->ptxData.data()));

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

    loadPTX();
}

detail::KernelStorage::KernelStorage(std::ifstream& i, logger_t logger)
    : logger(logger)
    , module(nullptr)
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

    loadPTX();
}

void detail::KernelStorage::loadPTX()
{
    logger->debug("Load module {}", this->machineName);

    //TEST:
    CUcontext ctx;
    CKL_SAFE_CALL(cuCtxGetCurrent(&ctx));
    std::thread::id threadId = std::this_thread::get_id();
    logger->debug("Current context: {}, current thread: {}", 
        reinterpret_cast<std::size_t>(ctx), 
        *static_cast<unsigned int*>(static_cast<void*>(&threadId))); //ugly casting, but otherwise I can't print

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
    if (infoLogData[0])
    {
        logger->debug("Load PTX log: {}", infoLog.get());
    }
    if (errorLog[0]) {
        logger->error("PTX loading error: {}", errorLog.get());
    }
    CKL_SAFE_CALL(err);

    //get cuda function and constants
    CKL_SAFE_CALL(cuModuleGetFunction(&this->function, this->module, this->machineName.data()));
    for (const auto& e : human2machine)
    {
        logger->debug("Fetch address for constant variable {} with machine name {}",
            e.first, e.second);
        CUdeviceptr addr;
        CKL_SAFE_CALL(cuModuleGetGlobal(&addr, nullptr, module, e.second.data()));
        constants[e.first] = addr;
        logger->debug("Constant variable {} has device pointer {}",
            e.first, addr);
    }

    CKL_SAFE_CALL(cuOccupancyMaxPotentialBlockSize(
        &minGridSize, &bestBlockSize, function, NULL, 0, 0));

    logger->debug("Module {} loaded successfully, block size: {}",
        this->machineName, bestBlockSize);
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
        logger->error("CUDA error {} when unloading module for kernel {}",
            pName, machineName);
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

void KernelFunction::callRaw(unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ, unsigned blockDimX,
    unsigned blockDimY, unsigned blockDimZ, unsigned sharedMemBytes, CUstream hStream, void** kernelParams)
{
    CUcontext ctx;
    CKL_SAFE_CALL(cuCtxGetCurrent(&ctx));
    ContextRAIID contextRaiid(ctx != ctx_ ? ctx_ : nullptr);

    CKL_SAFE_CALL(cuLaunchKernel(fun(), gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, nullptr));
}

static std::shared_ptr<spdlog::logger> get_logger(std::shared_ptr<spdlog::logger> logger)
{
    //1. check parent
    if (logger != nullptr) return logger;

    //2. does a logger already exist?
    logger = spdlog::get("ckl");
    if (logger != nullptr) return logger;

    //3. create new logger
    logger = spdlog::stdout_color_mt("ckl");
    assert(logger != nullptr);
    return logger;
}

KernelLoader::KernelLoader(std::shared_ptr<spdlog::logger> logger)
    : logger_(get_logger(logger))
{
    //query compute capability
    cudaDeviceProp props {0};
    auto retVal = cudaGetDeviceProperties(&props, 0);
    if (retVal == cudaErrorInsufficientDriver) {
        return;
    }
    CKL_SAFE_CALL(retVal);
    computeMajor_ = props.major;
    computeMinor_ = props.minor;
    logger_->info("Compiling kernels for device '{}' with compute capability {}.{}",
        props.name, props.major, props.minor);
    computeArchitecture_ = internal::Format::format("--gpu-architecture=compute_%d%d", computeMajor_, computeMinor_);

#ifdef CKL_NVCC_INCLUDE_DIR
    logger_->debug("NVCC include directory: {}", CKL_STR(CKL_NVCC_INCLUDE_DIR));
#else
    logger_->warn("No NVCC include directory specified. Compiling kernels will likely fail");
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
    logger_->debug("Current CUDA Driver context: {}", reinterpret_cast<std::size_t>(ctx_));
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

std::shared_ptr<spdlog::logger> KernelLoader::getLogger() const
{
    return logger_;
}

void KernelLoader::setLogLevel(spdlog::level::level_enum level)
{
    logger_->set_level(level);
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
    loadKernelCache();
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
    //check if we are in a multi-threaded environment
    CUcontext ctx;
    CKL_SAFE_CALL(cuCtxGetCurrent(&ctx));
    if (ctx_ == nullptr)
    {
        //no context found during initialization, set this one as the default one
        ctx_ = ctx;
    }
    if (ctx != ctx_)
    {
        logger_->debug("Attempt to call getKernel() from a different thread or different context than where the KernelLoader was created from. "
            "Expected context: {}, current context: {}. Restore the old context now.",
            reinterpret_cast<std::size_t>(ctx_), reinterpret_cast<std::size_t>(ctx));
    }
    ContextRAIID contextRaiid(ctx != ctx_ ? ctx_ : nullptr);

    loadKernelCache();

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
                constantNames, opts, logger_);

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
                logger_->error("Unable to compile kernel: {}", ex.what());
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
            logger_->error("Unable to create cache directory {}", absolute(cacheDirectory_).string());
            return;
        }
        else {
            logger_->info("Cache directory created at {}", absolute(cacheDirectory_).string());
        }
    }

    fs::path cacheFile = cacheDirectory_ / (includeFilesHash_ + ".kernel");
    std::ofstream o(cacheFile, std::ofstream::binary);
    if (!o.is_open())
    {
        logger_->error("Unable to open cache file {} for writing", cacheFile.string());
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
    logger_->debug("{} kernels written to the cache file", static_cast<int>(entrySize));
}

void KernelLoader::loadKernelCache()
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
        logger_->debug("Read from cache {}", cacheFile.string());
        std::ifstream i(cacheFile, std::ifstream::binary);
        if (!i.is_open())
        {
            logger_->error("Unable to open cache file {}", cacheFile.string());
            return;
        }
        unsigned int magic;
        i.read(reinterpret_cast<char*>(&magic), sizeof(int));
        if (magic != KERNEL_CACHE_MAGIC)
        {
            logger_->error("Invalid magic number, wrong file type or cache file is corrupted");
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

            const auto storage = std::make_shared<detail::KernelStorage>(i, logger_);
            kernelStorage_.emplace(kernelName, storage);
        }
        logger_->debug("{} kernels loaded from the cache", entrySize);
    }
}



CKL_NAMESPACE_END
