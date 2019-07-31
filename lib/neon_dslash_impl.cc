#include "neon_dslash_impl.h"
#include <cinttypes>
#include <tuple>

#include "neon_dslash_details.h"

namespace Chroma
{

// spinProjectDirMinus()
void decomp_plus(int lo, int hi, int id,
                 Spinor* spinorField, HalfSpinor* chi,
                 GaugeMat (*gauge)[4], int cb,
                 ShiftTable* sTab)
{
    int subgridVolCB = sTab->subgridVolCB();
    int low = subgridVolCB * cb + lo;
    int high = subgridVolCB * cb + hi;
    
    HalfSpinor* s3;
    HalfSpinor* s4;
    HalfSpinor* s5;
    HalfSpinor* s6;

    Spinor* sp;

    for (int idx = low; idx < high; ++idx) {
        int curSite = sTab->siteTable(idx);
        s3 = sTab->halfspinorBufferOffset(DECOMP_SCATTER, idx, 0);
        s4 = sTab->halfspinorBufferOffset(DECOMP_SCATTER, idx, 1);
        s5 = sTab->halfspinorBufferOffset(DECOMP_SCATTER, idx, 2);
        s6 = sTab->halfspinorBufferOffset(DECOMP_SCATTER, idx, 3);
        sp = &spinorField[curSite];

        decomp_gamma0_minus(*sp, *s3);
        decomp_gamma1_minus(*sp, *s4);
        decomp_gamma2_minus(*sp, *s5);
        decomp_gamma3_minus(*sp, *s6);
    }
}

// adj(gaugeMat) * spinProjectDirMinus
void decomp_hvv_plus(int lo, int hi, int id,
                     Spinor* spinorField, HalfSpinor* chi,
                     GaugeMat (*gaugeField)[4], int cb,
                     ShiftTable* sTab)
{
    GaugeMat* um1;
    GaugeMat* um2;
    GaugeMat* um3;
    GaugeMat* um4;

    HalfSpinor* s3;
    HalfSpinor* s4;
    HalfSpinor* s5;
    HalfSpinor* s6;

    int subgridVolCB = sTab->subgridVolCB();

    int low = cb * subgridVolCB + lo;
    int high = cb * subgridVolCB + hi;

    for (int idx = low; idx < high; ++idx) {
        int curSite = sTab->siteTable(idx);
        Spinor* sp = &spinorField[curSite];
                
        um1 = &gaugeField[curSite][0];
        um2 = &gaugeField[curSite][1];
        um3 = &gaugeField[curSite][2];
        um4 = &gaugeField[curSite][3];
        
        s3 = sTab->halfspinorBufferOffset(DECOMP_HVV_SCATTER, idx, 0);
        s4 = sTab->halfspinorBufferOffset(DECOMP_HVV_SCATTER, idx, 1);       
        s5 = sTab->halfspinorBufferOffset(DECOMP_HVV_SCATTER, idx, 2);
        s6 = sTab->halfspinorBufferOffset(DECOMP_HVV_SCATTER, idx, 3);
        
        decomp_hvv_gamma0_plus(*sp, *um1, *s3);
        decomp_hvv_gamma1_plus(*sp, *um2, *s4);
        decomp_hvv_gamma2_plus(*sp, *um3, *s5);
        decomp_hvv_gamma3_plus(*sp, *um4, *s6);
    }
}

void mvv_recons_plus(int lo, int hi, int id,
                     Spinor* spinorField, HalfSpinor* chi,
                     GaugeMat (*gaugeField)[4], int cb,
                     ShiftTable* sTab)
{
    GaugeMat* u1;
    GaugeMat* u2;
    GaugeMat* u3;
    GaugeMat* u4;

    HalfSpinor* hs1;
    HalfSpinor* hs2;
    HalfSpinor* hs3;
    HalfSpinor* hs4;

    int subgridVolCB = sTab->subgridVolCB();

    int low = cb * subgridVolCB + lo;
    int high = cb * subgridVolCB + hi;

    for (int idx = low; idx < high; ++idx) {
        int curSite = sTab->siteTable(idx);
        u1 = &gaugeField[curSite][0];
        u2 = &gaugeField[curSite][1];
        u3 = &gaugeField[curSite][2];
        u4 = &gaugeField[curSite][3];

        hs1 = sTab->halfspinorBufferOffset(RECONS_MVV_GATHER, idx, 0);
        hs2 = sTab->halfspinorBufferOffset(RECONS_MVV_GATHER, idx, 1);
        hs3 = sTab->halfspinorBufferOffset(RECONS_MVV_GATHER, idx, 2);
        hs4 = sTab->halfspinorBufferOffset(RECONS_MVV_GATHER, idx, 3);

        Spinor* sp = &spinorField[curSite];
        mvv_recons_4dir_minus(*hs1, *hs2, *hs3, *hs4,
                              *u1, *u2, *u3, *u4, *sp);
    }
}

void recons_plus(int lo, int hi, int id,
                 Spinor* spinorField, HalfSpinor* chi,
                 GaugeMat (*gaugeField)[4], int cb,
                 ShiftTable* sTab)
{
    HalfSpinor* hs1;
    HalfSpinor* hs2;
    HalfSpinor* hs3;
    HalfSpinor* hs4;
    Spinor* sp;

    int subgridVolCB = sTab->subgridVolCB();

    int low = cb * subgridVolCB + lo;
    int high = cb * subgridVolCB + hi;

    for (int idx = low; idx < high; ++idx) {
        int curSite = sTab->siteTable(idx);
        hs1 = sTab->halfspinorBufferOffset(RECONS_GATHER, idx, 0);
        hs2 = sTab->halfspinorBufferOffset(RECONS_GATHER, idx, 1);
        hs3 = sTab->halfspinorBufferOffset(RECONS_GATHER, idx, 2);
        hs4 = sTab->halfspinorBufferOffset(RECONS_GATHER, idx, 3);
        sp = &spinorField[curSite];

        recons_4dir_plus(*hs1, *hs2, *hs3, *hs4, *sp);
    }
}

void decomp_minus(int lo, int hi, int id,
                  Spinor* spinorField, HalfSpinor* chi,
                  GaugeMat (*gaugeField)[4], int cb,
                  ShiftTable* sTab)
{
    int subgridVolCB = sTab->subgridVolCB();
    int low = subgridVolCB * cb + lo;
    int high = subgridVolCB * cb + hi;
    
    HalfSpinor* s3;
    HalfSpinor* s4;
    HalfSpinor* s5;
    HalfSpinor* s6;

    Spinor* sp;

    for (int idx = low; idx < high; ++idx) {
        int curSite = sTab->siteTable(idx);
        s3 = sTab->halfspinorBufferOffset(DECOMP_SCATTER, idx, 0);
        s4 = sTab->halfspinorBufferOffset(DECOMP_SCATTER, idx, 1);
        s5 = sTab->halfspinorBufferOffset(DECOMP_SCATTER, idx, 2);
        s6 = sTab->halfspinorBufferOffset(DECOMP_SCATTER, idx, 3);
        sp = &spinorField[curSite];

        decomp_gamma0_plus(*sp, *s3);
        decomp_gamma1_plus(*sp, *s4);
        decomp_gamma2_plus(*sp, *s5);
        decomp_gamma3_plus(*sp, *s6);
    }
}

void decomp_hvv_minus(int lo, int hi, int id,
                      Spinor* spinorField, HalfSpinor* chi,
                      GaugeMat (*gaugeField)[4], int cb,
                      ShiftTable* sTab)
{
    GaugeMat* um1;
    GaugeMat* um2;
    GaugeMat* um3;
    GaugeMat* um4;

    HalfSpinor* s3;
    HalfSpinor* s4;
    HalfSpinor* s5;
    HalfSpinor* s6;

    int subgridVolCB = sTab->subgridVolCB();

    int low = cb * subgridVolCB + lo;
    int high = cb * subgridVolCB + hi;

    for (int idx = low; idx < high; ++idx) {
        int curSite = sTab->siteTable(idx);
        Spinor* sp = &spinorField[curSite];
                
        um1 = &gaugeField[curSite][0];
        um2 = &gaugeField[curSite][1];
        um3 = &gaugeField[curSite][2];
        um4 = &gaugeField[curSite][3];
        
        s3 = sTab->halfspinorBufferOffset(DECOMP_HVV_SCATTER, idx, 0);
        s4 = sTab->halfspinorBufferOffset(DECOMP_HVV_SCATTER, idx, 1);        
        s5 = sTab->halfspinorBufferOffset(DECOMP_HVV_SCATTER, idx, 2);
        s6 = sTab->halfspinorBufferOffset(DECOMP_HVV_SCATTER, idx, 3);
        
        decomp_hvv_gamma0_minus(*sp, *um1, *s3);
        decomp_hvv_gamma1_minus(*sp, *um2, *s4);
        decomp_hvv_gamma2_minus(*sp, *um3, *s5);
        decomp_hvv_gamma3_minus(*sp, *um4, *s6);        
    }    
}

void mvv_recons_minus(int lo, int hi, int id,
                      Spinor* spinorField, HalfSpinor* chi,
                      GaugeMat (*gaugeField)[4], int cb,
                      ShiftTable* sTab)
{
    GaugeMat* u1;
    GaugeMat* u2;
    GaugeMat* u3;
    GaugeMat* u4;

    HalfSpinor* hs1;
    HalfSpinor* hs2;
    HalfSpinor* hs3;
    HalfSpinor* hs4;

    int subgridVolCB = sTab->subgridVolCB();

    int low = cb * subgridVolCB + lo;
    int high = cb * subgridVolCB + hi;

    for (int idx = low; idx < high; ++idx) {
        int curSite = sTab->siteTable(idx);
        u1 = &gaugeField[curSite][0];
        u2 = &gaugeField[curSite][1];
        u3 = &gaugeField[curSite][2];
        u4 = &gaugeField[curSite][3];

        hs1 = sTab->halfspinorBufferOffset(RECONS_MVV_GATHER, idx, 0);
        hs2 = sTab->halfspinorBufferOffset(RECONS_MVV_GATHER, idx, 1);
        hs3 = sTab->halfspinorBufferOffset(RECONS_MVV_GATHER, idx, 2);
        hs4 = sTab->halfspinorBufferOffset(RECONS_MVV_GATHER, idx, 3);

        Spinor* sp = &spinorField[curSite];
        mvv_recons_4dir_plus(*hs1, *hs2, *hs3, *hs4,
                              *u1, *u2, *u3, *u4, *sp);
    }
}

void recons_minus(int lo, int hi, int id,
                  Spinor* spinorField, HalfSpinor* chi,
                  GaugeMat (*gaugeField)[4], int cb,
                  ShiftTable* sTab)
{
    HalfSpinor* hs1;
    HalfSpinor* hs2;
    HalfSpinor* hs3;
    HalfSpinor* hs4;
    Spinor* sp;

    int subgridVolCB = sTab->subgridVolCB();

    int low = cb * subgridVolCB + lo;
    int high = cb * subgridVolCB + hi;

    for (int idx = low; idx < high; ++idx) {
        int curSite = sTab->siteTable(idx);
        hs1 = sTab->halfspinorBufferOffset(RECONS_GATHER, idx, 0);
        hs2 = sTab->halfspinorBufferOffset(RECONS_GATHER, idx, 1);
        hs3 = sTab->halfspinorBufferOffset(RECONS_GATHER, idx, 2);
        hs4 = sTab->halfspinorBufferOffset(RECONS_GATHER, idx, 3);
        sp = &spinorField[curSite];

        recons_4dir_minus(*hs1, *hs2, *hs3, *hs4, *sp);
    }
}

} // namespace Chroma
    
