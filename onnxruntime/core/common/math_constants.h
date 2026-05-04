// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Portable math constants using C++20 <numbers>.
// POSIX constants like M_PI and M_SQRT2 are not part of the C++ standard and
// are not provided by all standard library implementations (e.g. libc++).
// This header defines them in terms of std::numbers so they are available
// on any conforming C++20 standard library.

#pragma once

#include <numbers>

#ifndef M_PI
#define M_PI std::numbers::pi
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 (1.0 / std::numbers::sqrt2)
#endif

#ifndef M_SQRT2
#define M_SQRT2 std::numbers::sqrt2
#endif
