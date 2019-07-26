#include "neon_dslash.h"
#include "neon_dslash_impl.h"

namespace Chroma
{


// Func should be stateless
template <typename Func>
void dispatchToThreads(Func func, int const nsites,
                       Spinor* spinorField, HalfSpinor* theHalfSpinor,
                       GaugeMat (*gaugeField)[4], int cb,
                       ShiftTable* stab)
{
    int nthreads;
    int id;
    int low;
    int high;

#pragma omp parallel shared(func, nsites, spinorField, theHalfSpinor, gaugeField, cb, stab) \
    private(id, nthreads, low, high)  default(none)
    {
        threads_num = omp_get_num_threads();
        nthreads = omp_get_thread_num();
        low = nsites * id / nthreads;
        high = nsites * (id+1) / nthreads;
        func(low, high, id, spinorField, theHalfSpinor, gaugeField, cb, stab);
    }
}


void NeonDslash::packGauge(multi1d<LatticeColorMatrix> const& gauge,
                          multi1d<GaugeMat>& packed /* out */)
{
    // assume packed is resized already
    size_t const sites = Layout::sitesOnNode();
    
    for (size_t i = 0; i < sites; ++i) {
        for (size_t d = 0; d < 4; ++d) {
            // packed[i + d] = gauge[d].elem(i);
            for (size_t n = 0; n < 3; ++n) {
                for (size_t m = 0; m < 3; ++m) {
                    packed[i+d][n][m][0] = gauge[d].elem(i).elem().elem(m, n).real();
                    packed[i+d][n][m][1] = gauge[d].elem(i).elem().elem(m, n).imag();
                }
            }
        }
    }
}



//! Full constructor with general coefficients
NeonDslash::NeonDslash(Handle<FermState<T,P,Q>> state,
                       const multi1d<Real>& coeffs_)   
{
    create(state, coeffs_);
}

//! Full constructor
NeonDslash::NeonDslash(Handle<FermState<T,P,Q>> state)
{ 
    create(state);
}
  
//! Full constructor with anisotropy
NeonDslash::NeonDslash(Handle<FermState<T,P,Q>> state,
                       const AnisoParam_t& aniso_) 
{

    create(state, aniso_);
}

//! Creation routine
void NeonDslash::create(Handle<FermState<T,P,Q>> state)
{
    multi1d<Real> cf(Nd);
    cf = 1.0;
    create(state, cf);
}

//! Creation routine with anisotropy
void NeonDslash::create(Handle<FermState<T,P,Q>> state,
                        const AnisoParam_t& anisoParam) 
{
    create(state, makeFermCoeffs(anisoParam)); 
}

//! Full constructor with general coefficients
void NeonDslash::create(Handle<FermState<T,P,Q>> state,
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
        QDPIO::cerr << "NeonDslash: error: fbc is null" << std::endl;
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

    dslashTable.reset(new DslashTable(subgrid));
    shiftTable.reset(new ShiftTable(subgrid,
                                    dslashTable->getChi1(), 
                                    dslashTable->getChi2(), 
                                    (HalfSpinor*(*)[4])(dslashTable->getRecvBufptr()), 
                                    (HalfSpinor*(*)[4])(dslashTable->getSendBufptr()),
                                    Layout::QDPXX_getSiteCoords,
                                    Layout::QDPXX_getLinearSiteIndex,
                                    Layout::QDPXX_nodeNumber
                         ));
}

void NeonDslash::apply(T& chi, const T& psiArg, PlusMinus isign, int cb) const
{
    START_CODE();
    GaugeMat (*u)[4] = (GaugeMat(*)[4]) &packedGauge[0];
    Spinor* psi = (Spinor*) &psiArg.elem(0);
    HalfSpinor* res = (Spinor*) &chi.elem(0);
    
    HalfSpinor* chi1 = tab->getChi1();
    HalfSpinor* chi2 = tab->getChi2();
    int subgrid_vol_cb = s_tab->subgridVolCB();

    int sourceCB = 1 - cb;
    
    if (isign == 1) {

        tab->startReceives();

        dispatchToThreads(decomp_plus,
                          psi,
                          chi1,
                          u,
                          s_tab.get(),
                          sourceCB,
                          subgrid_vol_cb);

        tab->startSendForward(); 
        dispatchToThreads(decomp_hvv_plus,
                          psi,
                          chi2,
                          u,
                          s_tab.get(),
                          sourceCB,
                          subgrid_vol_cb);

        tab->finishSendForward();
        tab->finishReceiveFromBack();
        tab->startSendBack();

        dispatchToThreads(mvv_recons_plus,
                          res,
                          chi1,
                          u,
                          s_tab.get(),
                          1-sourceCB,
                          subgrid_vol_cb);

        tab->finishSendBack();
        tab->finishReceiveFromForward();    

        dispatchToThreads(recons_plus,
                          res, 
                          chi2,
                          u,	
                          s_tab.get(),
                          1-sourceCB,
                          subgrid_vol_cb);

    } else if (isign == -1) {
        
        tab->startReceives();

        dispatchToThreads(decomp_minus,
                          psi,
                          chi1,
                          u,
                          s_tab.get(),
                          sourceCB,
                          subgrid_vol_cb);

        tab->startSendForward(); 
        dispatchToThreads(decomp_hvv_minus,
                          psi,
                          chi2,
                          u,
                          s_tab.get(),
                          sourceCB,
                          subgrid_vol_cb);

        tab->finishSendForward();
        tab->finishReceiveFromBack();
        tab->startSendBack();

        dispatchToThreads(mvv_recons_minus,
                          res,
                          chi1,
                          u,
                          s_tab.get(),
                          1-sourceCB,
                          subgrid_vol_cb);

        tab->finishSendBack();
        tab->finishReceiveFromForward();    

        dispatchToThreads(recons_minus,
                          res, 
                          chi2,
                          u,	
                          s_tab.get(),
                          1-sourceCB,
                          subgrid_vol_cb);
    } else {
        // not possible
        fprintf(stderr, "isign == %d\n", (int)isign);
        QMP_abort(1);
    }

    getFermBC().modifyF(chi, QDP::rb[cb]);
    END_CODE();
}



} // namespace Chroma
