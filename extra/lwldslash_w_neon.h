#ifndef LWLDSLASH_W_NEON_H
#define LWLDSLASH_W_NEON_H

#include "state.h"
#include "io/aniso_io.h"
#include "actions/ferm/linop/lwldslash_base_w.h"

#include "neon_dslash.h"

namespace Chroma
{

class NeonWilsonDslash : public WilsonDslashBase<LatticeFermion,
                                           multi1d<LatticeColorMatrix>, 
                                           multi1d<LatticeColorMatrix>>
{
public:
    using T = LatticeFermion;
    using P = multi1d<LatticeColorMatrix>;
    using Q = multi1d<LatticeColorMatrix>;

    //! Empty constructor. Must use create later
    NeonWilsonDslash() = default;

    //! Full constructor
    NeonWilsonDslash(Handle<FermState<T,P,Q>> state);

    //! Full constructor with anisotropy
    NeonWilsonDslash(Handle<FermState<T,P,Q>> state,
               const AnisoParam_t& aniso_);

    //! Full constructor with general coefficients
    NeonWilsonDslash(Handle<FermState<T,P,Q>> state,
               const multi1d<Real>& coeffs_);

    //! Creation routine
    void create(Handle<FermState<T,P,Q>> state);

    //! Creation routine with anisotropy
    void create(Handle<FermState<T,P,Q>> state,
		const AnisoParam_t& aniso_);

    //! Full constructor with general coefficients
    void create(Handle< FermState<T,P,Q> > state, 
		const multi1d<Real>& coeffs_);

    void apply(T& chi, const T& psi, PlusMinus isign, int cb) const {
        impl.apply((float*)chi.getF(), (float*)psi.getF(), isign, cb); // use the OLattice backdoor
    }

    //! Return the fermion BC object for this linear operator
    const FermBC<T,P,Q>& getFermBC() const {return *fbc;}

protected:

    //! Get the anisotropy parameters
    const multi1d<Real>& getCoeffs() const {return coeffs;}    

private:
    NeonDslash impl; // the real implementation
    
    multi1d<Real> coeffs;
    Handle<FermBC<T,P,Q>> fbc;
    multi1d<GaugeMat> packedGauge;
    
    static void packGauge(multi1d<LatticeColorMatrix> const& gauge,
                          multi1d<GaugeMat>& /*out*/ packed);
};

} // namespace Chroma

#endif // LWLDSLASH_W_NEON_H
