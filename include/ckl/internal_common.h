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

#define CKL_NAMESPACE_BEGIN namespace ckl {
#define CKL_NAMESPACE_END }

#define CKL_STR_DETAIL(x) #x
#define CKL_STR(x) CKL_STR_DETAIL(x)
