
#include "lwldslash_w_neon.h"

namespace Chroma
{

void NeonWilsonDslash::packGauge(multi1d<LatticeColorMatrix> const& gauge,
                          multi1d<GaugeMat>& packed /* out */)
{
    // assume packed is resized already
    size_t const sites = Layout::sitesOnNode();
    
    for (size_t i = 0; i < sites; ++i) {
        for (size_t d = 0; d < 4; ++d) {
            // packed[i + d] = gauge[d].elem(i);
            for (size_t n = 0; n < 3; ++n) {
                for (size_t m = 0; m < 3; ++m) {
                    packed[i*4+d][n][m][0] = gauge[d].elem(i).elem().elem(m, n).real();
                    packed[i*4+d][n][m][1] = gauge[d].elem(i).elem().elem(m, n).imag();
                }
            }
        }
    }
}



//! Full constructor with general coefficients
NeonWilsonDslash::NeonWilsonDslash(Handle<FermState<T,P,Q>> state,
                       const multi1d<Real>& coeffs_)   
{
    create(state, coeffs_);
}

//! Full constructor
NeonWilsonDslash::NeonWilsonDslash(Handle<FermState<T,P,Q>> state)
{ 
    create(state);
}
  
//! Full constructor with anisotropy
NeonWilsonDslash::NeonWilsonDslash(Handle<FermState<T,P,Q>> state,
                       const AnisoParam_t& aniso_) 
{

    create(state, aniso_);
}

//! Creation routine
void NeonWilsonDslash::create(Handle<FermState<T,P,Q>> state)
{
    multi1d<Real> cf(Nd);
    cf = 1.0;
    create(state, cf);
}

//! Creation routine with anisotropy
void NeonWilsonDslash::create(Handle<FermState<T,P,Q>> state,
                        const AnisoParam_t& anisoParam) 
{
    create(state, makeFermCoeffs(anisoParam)); 
}

//! Full constructor with general coefficients
void NeonWilsonDslash::create(Handle<FermState<T,P,Q>> state,
                        const multi1d<Real>& coeffs_)
{
    // the final entry:
    
    // Save a copy of the aniso params original fields and with aniso folded in
    coeffs = coeffs_;

    // Save a copy of the fermbc
    fbc = state->getFermBC();

    // Sanity check
    if (fbc.operator->() == 0)
    {
        QDPIO::cerr << "NeonWilsonDslash: error: fbc is null" << std::endl;
        QDP_abort(1);
    }

    // Fold in anisotropy
    multi1d<LatticeColorMatrix> u = state->getLinks();
  
    // Rescale the u fields by the anisotropy
    for(int mu=0; mu < u.size(); ++mu)
    {
        u[mu] *= coeffs[mu];
    }

    // Pack the gauge fields
    packedGauge.resize(Nd * Layout::sitesOnNode());
    
    packGauge(u, packedGauge);

    int subgrid[4];
    const auto& subgridNRow = Layout::subgridLattSize();
    subgrid[0] = subgridNRow[0];
    subgrid[1] = subgridNRow[1];
    subgrid[2] = subgridNRow[2];
    subgrid[3] = subgridNRow[3];
    
    impl.create(subgrid, packedGauge.slice(),
                Layout::QDPXX_getSiteCoords,
                Layout::QDPXX_getLinearSiteIndex,
                Layout::QDPXX_nodeNumber);
}

} // namespace Chroma
