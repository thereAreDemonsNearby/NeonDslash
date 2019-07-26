#ifndef NEON_DSLASH_H
#define NEON_DSLASH_H

#include <memory>
#include "state.h"
#include "io/aniso_io.h"
#include "actions/ferm/linop/lwldslash_base_w.h"

#include "neon_dslash_types.h"
#include "shift_table.h"
#include "dslash_table.h"

namespace Chroma
{

// single floating point only
class NeonDslash : public WilsonDslashBase<LatticeFermion,
                                           multi1d<LatticeColorMatrix>, 
                                           multi1d<LatticeColorMatrix>>
{
public:
    using T = LatticeFermion;
    using P = multi1d<LatticeColorMatrix>;
    using Q = multi1d<LatticeColorMatrix>;

    //! Empty constructor. Must use create later
    NeonDslash() = delete;

    //! Full constructor
    NeonDslash(Handle<FermState<T,P,Q>> state);

    //! Full constructor with anisotropy
    NeonDslash(Handle<FermState<T,P,Q>> state,
               const AnisoParam_t& aniso_);

    //! Full constructor with general coefficients
    NeonDslash(Handle<FermState<T,P,Q>> state,
               const multi1d<Real>& coeffs_);

    //! Creation routine
    void create(Handle<FermState<T,P,Q>> state);

    //! Creation routine with anisotropy
    void create(Handle<FermState<T,P,Q>> state,
		const AnisoParam_t& aniso_);

    //! Full constructor with general coefficients
    void create(Handle< FermState<T,P,Q> > state, 
		const multi1d<Real>& coeffs_);

    void apply(T& chi, const T& psi, PlusMinus isign, int cb) const;

    //! Return the fermion BC object for this linear operator
    const FermBC<T,P,Q>& getFermBC() const {return *fbc;}

private:
    multi1d<Real> coeffs;
    Handle<FermBC<T,P,Q>> fbc;
    multi1d<GaugeMat> packedGauge;

    // extra needed:
    std::unique_ptr<DslashTable> dslashTable;
    std::unique_ptr<ShiftTable> shiftTable;
    

    //! Get the anisotropy parameters
    const multi1d<Real>& getCoeffs() const {return coeffs;}

    static void packGauge(multi1d<LatticeColorMatrix> const& gauge,
                          multi1d<GaugeMat>& /*out*/ packed);
};

} // namespace Chroma

#endif // NEONDSLASH_H
