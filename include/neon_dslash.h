#ifndef NEON_DSLASH_H
#define NEON_DSLASH_H

#include <memory>

#include "neon_dslash_types.h"
#include "shift_table.h"
#include "dslash_table.h"

namespace Chroma
{

// single floating point only
class NeonDslash
{
public:

    //! Empty constructor. Must use create later
    NeonDslash() = default;
    void create(int subgrid[], /* int subgrid[4] */
                GaugeMat* packedGauge,
                void (*getSiteCoords)(int coord[], int node, int linear),
                int (*getLinearSiteIndex)(const int coord[]),
                int (*nodeNumber)(const int coord[]));
    
    void apply(float* chi, float* psi, int isign, int cb) const;


private:
    GaugeMat* packedGauge; // only a view. not owned.

    // extra needed:
    std::unique_ptr<DslashTable> dslashTable;
    std::unique_ptr<ShiftTable> shiftTable;   
};

} // namespace Chroma

#endif // NEONDSLASH_H
