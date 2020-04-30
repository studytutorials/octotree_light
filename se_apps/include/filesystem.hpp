/*
 * SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Detect std::filesystem support.

// Proper std::filesystem support.
#if        (defined(__GNUC__)        && __GNUC__        >= 8) \
        || (defined(__clang_major__) && __clang_major__ >= 7) \
        || (defined(_MSC_VER)        && _MSC_VER        >= 1914)
#define __HAS_FILESYSTEM
// Experimental std::filesystem support.
#elif      (defined(__GNUC__)        && __GNUC__        >= 6) \
        || (defined(__clang_major__) && __clang_major__ >= 6)
#define __HAS_EXP_FILESYSTEM
// No std::filesystem support.
#else
#error A compiler with support for std::filesystem is required
#endif

// Include the appropriate header.
#if defined(__HAS_FILESYSTEM)
#include <filesystem>
#elif defined(__HAS_EXP_FILESYSTEM)
#include <experimental/filesystem>
#endif

// Alias namespace.
#if defined(__HAS_FILESYSTEM)
namespace stdfs = std::filesystem;
#elif defined(__HAS_EXP_FILESYSTEM)
namespace stdfs = std::experimental::filesystem;
#endif


