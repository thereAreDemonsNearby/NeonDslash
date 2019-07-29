#ifndef NEON_DSLASH_TYPES_H
#define NEON_DSLASH_TYPES_H

#include <cstddef>

namespace Chroma
{
// some primitive types:
using Spinor = float[4][3][2]; 
using HalfSpinor = float[3][2][2]; // transposed to fit 4-lane simd
using GaugeMat = float[3][3][2];

namespace Cache
{
constexpr size_t CacheLineSize = 64;
constexpr size_t CacheSetSize = 16*1024;
}
} // end namespace Chroma


#endif // NEONDSLASH_TYPES_H
