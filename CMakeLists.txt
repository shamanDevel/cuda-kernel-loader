# CMakeLists.txt -- Build system for the cuda-kernel-loader (ckl) modules
#
# Copyright (c) 2022 Sebastian Weiss <sebastian13.weiss@tum.de>
#
# All rights reserved. Use of this source code is governed by a
# MIT-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.4)

# The `cmake_minimum_required(VERSION 3.4...3.18)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.18)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.18)
endif()
cmake_policy(SET CMP0127 NEW)

# Extract project version from source
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/ckl/common.h"
     ckl_version_defines REGEX "#define CKL_VERSION_(MAJOR|MINOR|PATCH) ")

foreach(ver ${ckl_version_defines})
  if(ver MATCHES [[#define CKL_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$]])
    set(CKL_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
  endif()
endforeach()

if(CKL_VERSION_PATCH MATCHES [[\.([a-zA-Z0-9]+)$]])
  set(ckl_VERSION_TYPE "${CMAKE_MATCH_1}")
endif()
string(REGEX MATCH "^[0-9]+" CKL_VERSION_PATCH "${CKL_VERSION_PATCH}")

project(
  ckl
  LANGUAGES CXX
  VERSION "${CKL_VERSION_MAJOR}.${CKL_VERSION_MINOR}.${CKL_VERSION_PATCH}")

# Standard includes
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)

if(NOT ckl_FIND_QUIETLY)
  message(STATUS "ckl v${ckl_VERSION} ${ckl_VERSION_TYPE}")
endif()

# Avoid infinite recursion if tests include this as a subdirectory
if(DEFINED CKL_MASTER_PROJECT)
  set(CKL_TEST OFF)
endif()

# Check if ckl is being used directly or via add_subdirectory
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND NOT DEFINED CKL_MASTER_PROJECT)
  ### Warn if not an out-of-source builds
  if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    set(lines
        "You are building in-place. If that is not what you intended to "
        "do, you can clean the source directory with:\n"
        "rm -r CMakeCache.txt CMakeFiles/ cmake_uninstall.cmake cklConfig.cmake "
        "cklConfigVersion.cmake tests/CMakeFiles/\n")
    message(AUTHOR_WARNING ${lines})
  endif()

  set(CKL_MASTER_PROJECT ON)

  if(OSX AND CMAKE_VERSION VERSION_LESS 3.7)
    # Bug in macOS CMake < 3.7 is unable to download catch
    message(WARNING "CMAKE 3.7+ needed on macOS to download catch, and newer HIGHLY recommended")
  elseif(WINDOWS AND CMAKE_VERSION VERSION_LESS 3.8)
    # Only tested with 3.8+ in CI.
    message(WARNING "CMAKE 3.8+ tested on Windows, previous versions untested")
  endif()

  message(STATUS "CMake ${CMAKE_VERSION}")

  if(CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()

  set(ckl_system "")
else()
  set(CKL_MASTER_PROJECT OFF)
  set(ckl_system SYSTEM)
endif()

# Options
option(CKL_TEST "Build ckl test suite?" ${CKL_MASTER_PROJECT})
option(CKL_SHARED "Build ckl as a shared library?" OFF)

# CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
    STRING(LENGTH "${ARCH_FLAGS}" ckl_ARCH_FLAGS_LENGTH)
    MATH(EXPR ckl_ARCH_FLAGS_LENGTH2 "${ckl_ARCH_FLAGS_LENGTH} - 2")
    STRING(SUBSTRING "${ARCH_FLAGS}" ${ckl_ARCH_FLAGS_LENGTH2} -1 ckl_ARCH_FLAGS2)
    set(ckl_ARCH_FLAGS3 "--gpu-architecture=compute_${ckl_ARCH_FLAGS2}")
    set(ckl_requires_arch_flags OFF)
else()
    set(ckl_requires_arch_flags ON)
endif()

cmake_dependent_option(CKL_ARCH_FLAGS "CUDA Architecture specification" "75" "ckl_requires_arch_flags" ${ckl_ARCH_FLAGS3})

list(APPEND ckl_CUDA_NVCC_FLAGS ${CKL_ARCH_FLAGS} "-std=c++17" "--use_fast_math" "--generate-line-info" "-Xptxas" "-v")
message(STATUS "ckl compilation flags: ${ckl_CUDA_NVCC_FLAGS}")

foreach(t ${ckl_CUDA_NVCC_FLAGS})
    set(CKL_NVCC_ARGS_C "${CKL_NVCC_ARGS_C}   \"${t}\",\n")
endforeach()
configure_file(src/ckl_nvcc_args.inl.in ckl_nvcc_args.inl)

# Files
set(CKL_HEADERS
    include/ckl/common.h
    include/ckl/kernel_loader.h
    include/ckl/errors.h
    include/ckl/cmrc_loader.h)
    
set(CKL_SOURCES
    src/kernel_loader.cpp
    src/sha1.h
    src/sha1.cpp)

# Compare with grep and warn if mismatched
if(CKL_MASTER_PROJECT AND NOT CMAKE_VERSION VERSION_LESS 3.12)
  file(
    GLOB_RECURSE _ckl_header_check
    LIST_DIRECTORIES false
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    CONFIGURE_DEPENDS "include/ckl/*.h")
  set(_ckl_here_only ${CKL_HEADERS})
  set(_ckl_disk_only ${_ckl_header_check})
  list(REMOVE_ITEM _ckl_here_only ${_ckl_header_check})
  list(REMOVE_ITEM _ckl_disk_only ${CKL_HEADERS})
  if(_ckl_here_only)
    message(AUTHOR_WARNING "CKL_HEADERS has extra files:" ${_ckl_here_only})
  endif()
  if(_ckl_disk_only)
    message(AUTHOR_WARNING "CKL_HEADERS is missing files:" ${_ckl_disk_only})
  endif()
endif()

# CMake 3.12 added list(TRANSFORM <list> PREPEND
# But we can't use it yet
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/" CKL_HEADERS
               "${CKL_HEADERS}")
string(REPLACE "src/" "${CMAKE_CURRENT_SOURCE_DIR}/src/" CKL_SOURCES
               "${CKL_SOURCES}")
string(REPLACE "tests/" "${CMAKE_CURRENT_SOURCE_DIR}/tests/" CKL_TESTS
               "${CKL_TESTS}")

# Cache variable so this can be used in parent projects
set(ckl_INCLUDE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/include"
    CACHE INTERNAL "Directory where ckl headers are located")

# Backward compatible variable for add_subdirectory mode
if(NOT CKL_MASTER_PROJECT)
  set(CKL_INCLUDE_DIR
      "${ckl_INCLUDE_DIR}"
      CACHE INTERNAL "")
endif()

# Note: when creating targets, you cannot use if statements at configure time -
# you need generator expressions, because those will be placed in the target file.
# You can also place ifs *in* the Config.in, but not here.

# This section builds targets
# Non-IMPORT targets cannot be defined twice
if(NOT TARGET ckl)
  # Build the headers-only target (no Python included):
  # (long name used here to keep this from clashing in subdirectory mode)
  if (CKL_SHARED)
    add_library(ckl SHARED
        ${CKL_HEADERS} ${CKL_SOURCES})
  else()
    add_library(ckl STATIC
        ${CKL_HEADERS} ${CKL_SOURCES})
  endif()
  add_library(ckl::ckl ALIAS ckl) # to match exported target

  set_target_properties(ckl PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )

  target_include_directories(
    ckl ${ckl_system} PUBLIC ${ckl_INCLUDE_DIR})
  target_include_directories(ckl PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  # CUDA Stuff
  target_link_libraries(ckl PRIVATE
    CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
  target_compile_definitions(ckl PRIVATE 
    CKL_NVCC_INCLUDE_DIR=${CUDA_INCLUDE_DIRS})
  
else()
  # It is invalid to install a target twice, too.
  set(CKL_INSTALL OFF)
endif()


# BUILD_TESTING takes priority, but only if this is the master project
if(CKL_MASTER_PROJECT AND DEFINED BUILD_TESTING)
  if(BUILD_TESTING)
    add_subdirectory(tests)
  endif()
else()
  if(CKL_TEST)
    add_subdirectory(tests)
  endif()
endif()

# Better symmetry with find_package(ckl CONFIG) mode.
if(NOT CKL_MASTER_PROJECT)
  set(ckl_FOUND
      TRUE
      CACHE INTERNAL "True if ckl and all required components found on the system")
endif()
