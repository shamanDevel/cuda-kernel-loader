/*
    cuda-kernel-loader/internal_common.h -- Basic macros

    Copyright (c) 2022 Sebastian Weiss <sebastian13.weiss@tum.de>

    All rights reserved. Use of this source code is governed by a
    MIT-style license that can be found in the LICENSE file.
*/

#pragma once

#define CKL_VERSION_MAJOR 1
#define CKL_VERSION_MINOR 0
#define CKL_VERSION_PATCH 0

#define CKL_NAMESPACE ckl
#define CKL_NAMESPACE_BEGIN namespace ckl {
#define CKL_NAMESPACE_END }

#define CKL_STR_DETAIL(x) #x
#define CKL_STR(x) CKL_STR_DETAIL(x)

#ifndef CKL_ALWAYS_SYNC
/**
 * If set to true via preprocess macros or so,
 * issues a cudaDeviceSynchronize() after every kernel launch or cuda function.
 * This help in pinning down errors, but drastically slows down the execution.
 */
#define CKL_ALWAYS_SYNC 0
#endif

/**
* \brief Returns the integer division x/y rounded up.
* Taken from https://stackoverflow.com/a/2745086/4053176
*/
#define CKL_DIV_UP(x, y) (((x) + (y) - 1) / (y))
