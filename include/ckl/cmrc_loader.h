#pragma once

/**
 * IMPORTANT:
 * This file requires the CMRC resource compiler.
 * See https://github.com/vector-of-bool/cmrc
 * Include and use this header only if you want to use this
 *  embedded file system
 */

#include <cmrc/cmrc.hpp>
#include "kernel_loader.h"

CKL_NAMESPACE_BEGIN

/**
 * File loader that loads all files from the embedded filesystem using
 * the CMRC resource compiler.
 * See: https://github.com/vector-of-bool/cmrc
 *
 * Usage:
 * <code>
 * auto fs = cmrc::<my-lib-ns>::get_filesystem();
 * auto loader = std::make_shared<CMRCLoader>(fs);
 * </code>
 */
class CMRCLoader : public IFileLoader
{
    const cmrc::embedded_filesystem fs_;
public:
    explicit CMRCLoader(const cmrc::embedded_filesystem& fs)
        : fs_(fs)
    {
    }

private:
    void populateRecursive(std::vector<NameAndContent>& files,
        const cmrc::directory_entry& e, const std::string& currentPath)
    {
        if (e.is_file())
        {
            auto f = fs_.open(currentPath + e.filename());
            std::string content(f.size(), '\0');
            memcpy(content.data(), f.begin(), f.size());
            files.push_back({ currentPath + e.filename(), content });
        }
        else
        {
            for (const auto& e2 : fs_.iterate_directory(currentPath + e.filename()))
                populateRecursive(files, e2, currentPath + e.filename() + "/");
        }
    }

public:
    void populate(std::vector<NameAndContent>& files) override
    {
        for (const auto& e : fs_.iterate_directory(""))
            populateRecursive(files, e, "");
    }
};

CKL_NAMESPACE_END
