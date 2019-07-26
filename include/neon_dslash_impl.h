#ifndef NEON_DSLASH_IMPL_H
#define NEON_DSLASH_IMPL_H

#include <arm_neon.h>

#include "shift_table.h"
#include "dslash_table.h"
#include "neon_dslash_types.h"

namespace Chroma
{

void decomp_plus(int lo, int hi, int id,
                 Spinor* spinorField, HalfSpinor* chi,
                 GaugeMat (*gaugeField)[4], int cb,
                 ShiftTable* sTab);

void decomp_hvv_plus(int lo, int hi, int id,
                     Spinor* sp, HalfSpinor* chi,
                     GaugeMat (*gauge)[4], int cb,
                     ShiftTable* sTab);

void mvv_recons_plus(int lo, int hi, int id,
                     Spinor* sp, HalfSpinor* chi,
                     GaugeMat (*gauge)[4], int cb,
                     ShiftTable* sTab);

void recons_plus(int lo, int hi, int id,
                 Spinor* sp, HalfSpinor* chi,
                 GaugeMat (*gauge)[4], int cb,
                 ShiftTable* sTab);

void decomp_minus(int lo, int hi, int id,
                  Spinor* sp, HalfSpinor* chi,
                  GaugeMat (*gauge)[4], int cb,
                  ShiftTable* sTab);

void decomp_hvv_minus(int lo, int hi, int id,
                      Spinor* sp, HalfSpinor* chi,
                      GaugeMat (*gauge)[4], int cb,
                      ShiftTable* sTab);

void mvv_recons_minus(int lo, int hi, int id,
                      Spinor* sp, HalfSpinor* chi,
                      GaugeMat (*gauge)[4], int cb,
                      ShiftTable* sTab);

void recons_minus(int lo, int hi, int id,
                  Spinor* sp, HalfSpinor* chi,
                  GaugeMat (*gauge)[4], int cb,
                  ShiftTable* sTab);

} // namespace Chroma

#endif NEON_DSLASH_IMPL_H
