# CMakeLists.txt -- Build system for the ckl test suite
#
# Copyright (c) 202 Sebastian Weiss <sebastian13.weiss@tum.de>
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


# Full set of test files
set(CKL_KERNEL_FILES
    kernels/axpy.cuh
    kernels/unary.cuh
    kernels/constants.cuh
    )
set(CKL_TEST_FILES
    main.cpp
    catch.hpp
    test_code_generation.cpp
    test_loaders.cpp
    test_constants.cpp
    )

add_executable(ckl-tests
    ${CKL_TEST_FILES} ${CKL_KERNEL_FILES})
    
set_target_properties(ckl-tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(ckl-tests
    PRIVATE ckl::ckl CUDA::cudart)

# Compile with compiler warnings turned on
if(MSVC)
    target_compile_options(ckl-tests PRIVATE /W4)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Intel|Clang)" AND NOT CKL_CUDA_TESTS)
    target_compile_options(
      ckl-tests
      PRIVATE -Wall
              -Wextra
              -Wconversion
              -Wcast-qual
              -Wdeprecated
              -Wundef
              -Wnon-virtual-dtor)
endif()

# embedded kernels
file(DOWNLOAD "https://raw.githubusercontent.com/vector-of-bool/cmrc/master/CMakeRC.cmake"
				"${CMAKE_BINARY_DIR}/CMakeRC.cmake")
include("${CMAKE_BINARY_DIR}/CMakeRC.cmake")
cmrc_add_resource_library(
	ckl-tests-kernels

	ALIAS ckl-tests::rc
	NAMESPACE kernels
    WHENCE kernels

	${CKL_KERNEL_FILES}
	)
target_link_libraries(ckl-tests PRIVATE ckl-tests::rc)
