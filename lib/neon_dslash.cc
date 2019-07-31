#include <omp.h>

#include "neon_dslash.h"
#include "neon_dslash_impl.h"

namespace Chroma
{


// Func should be stateless
template <typename Func>
void dispatchToThreads(Func func,
                       Spinor* spinorField, HalfSpinor* theHalfSpinor,
                       GaugeMat (*gaugeField)[4],
                       ShiftTable* stab, int cb, int const nsites)
{
    int nthreads;
    int id;
    int low;
    int high;

#pragma omp parallel shared(func, spinorField, theHalfSpinor, gaugeField, cb, stab) \
    private(id, nthreads, low, high) default(none)
    {
        nthreads = omp_get_num_threads();
        id = omp_get_thread_num();
        low = nsites * id / nthreads;
        high = nsites * (id+1) / nthreads;
        func(low, high, id, spinorField, theHalfSpinor, gaugeField, cb, stab);
    }
}

//! Full constructor with general coefficients
void NeonDslash::create(int subgrid[], /* int subgrid[4] */
                        GaugeMat* gauge,
                        void (*getSiteCoords)(int coord[], int node, int linear),
                        int (*getLinearSiteIndex)(const int coord[]),
                        int (*nodeNumber)(const int coord[]))
{
    packedGauge = gauge;
    
    dslashTable.reset(new DslashTable(subgrid));
    shiftTable.reset(new ShiftTable(subgrid,
                                    dslashTable->getChi1(),
                                    dslashTable->getChi2(), 
                                    (HalfSpinor*(*)[4])(dslashTable->getRecvBufptr()),
                                    (HalfSpinor*(*)[4])(dslashTable->getSendBufptr()),
                                    getSiteCoords,
                                    getLinearSiteIndex,
                                    nodeNumber
                         ));
}

void NeonDslash::apply(float* chi, float* psiArg, int isign, int cb) const
{
    GaugeMat (*u)[4] = (GaugeMat(*)[4]) &packedGauge[0];
    Spinor* psi = (Spinor*) psiArg;
    Spinor* res = (Spinor*) chi;
    
    HalfSpinor* chi1 = dslashTable->getChi1();
    HalfSpinor* chi2 = dslashTable->getChi2();
    int subgrid_vol_cb = shiftTable->subgridVolCB();

    int sourceCB = 1 - cb;
    
    if (isign == 1) {

        dslashTable->startReceives();
        
        dispatchToThreads(decomp_plus,
                          psi,
                          chi1,
                          u,
                          shiftTable.get(),
                          sourceCB,
                          subgrid_vol_cb);

        dslashTable->startSendForward(); 
        dispatchToThreads(decomp_hvv_plus,
                          psi,
                          chi2,
                          u,
                          shiftTable.get(),
                          sourceCB,
                          subgrid_vol_cb);

        dslashTable->finishSendForward();
        dslashTable->finishReceiveFromBack();
        dslashTable->startSendBack();

        dispatchToThreads(mvv_recons_plus,
                          res,
                          chi1,
                          u,
                          shiftTable.get(),
                          1-sourceCB,
                          subgrid_vol_cb);

        dslashTable->finishSendBack();
        dslashTable->finishReceiveFromForward();    

        dispatchToThreads(recons_plus,
                          res, 
                          chi2,
                          u,	
                          shiftTable.get(),
                          1-sourceCB,
                          subgrid_vol_cb);

    } else if (isign == -1) {
        
        dslashTable->startReceives();

        dispatchToThreads(decomp_minus,
                          psi,
                          chi1,
                          u,
                          shiftTable.get(),
                          sourceCB,
                          subgrid_vol_cb);

        dslashTable->startSendForward(); 
        dispatchToThreads(decomp_hvv_minus,
                          psi,
                          chi2,
                          u,
                          shiftTable.get(),
                          sourceCB,
                          subgrid_vol_cb);

        dslashTable->finishSendForward();
        dslashTable->finishReceiveFromBack();
        dslashTable->startSendBack();

        dispatchToThreads(mvv_recons_minus,
                          res,
                          chi1,
                          u,
                          shiftTable.get(),
                          1-sourceCB,
                          subgrid_vol_cb);

        dslashTable->finishSendBack();
        dslashTable->finishReceiveFromForward();

        dispatchToThreads(recons_minus,
                          res, 
                          chi2,
                          u,	
                          shiftTable.get(),
                          1-sourceCB,
                          subgrid_vol_cb);
    } else {
        // not possible
        throw 0;
    }
}



} // namespace Chroma
