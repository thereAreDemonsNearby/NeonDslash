#ifndef NEON_DSLASH_DETAILS_H
#define NEON_DSLASH_DETAILS_H

// header only library
#include <arm_neon.h>
#include "neon_dslash_types.h"

namespace
{
using Chroma::GaugeMat;
using Chroma::Spinor;
using Chroma::HalfSpinor;

inline void reverse_real_img(float32x4_t& vec1, float32x4_t& vec2, float32x4_t& vec3)
{
    vec1 = vrev64q_f32(vec1);
    vec2 = vrev64q_f32(vec2);
    vec3 = vrev64q_f32(vec3);
}

inline void reverse_elements(float32x4_t& v1, float32x4_t& v2, float32x4_t& v3)
{
    // swap lower half and higher half
    v1 = vextq_f32(v1, v1, 2);
    v2 = vextq_f32(v2, v2, 2);
    v3 = vextq_f32(v3, v3, 2);
    reverse_real_img(v1, v2, v3);
}

// hope it's inlined
inline void swizzle(float32x4_t& vec1, float32x4_t& vec2, float32x4_t& vec3)
{
    // original:
    // vec1: 00re 00img 01re 01img
    // vec2: 02re 02img 10re 10img
    // vec3: 11re 11img 12re 12img

    // aim:
    // vec1: 00re 00img 10re 10img
    // vec2: 01re 01img 11re 11img
    // vec3: 02re 02img 12re 12img
        
    float32x4_t t1 = vextq_f32(vec2, vec1, 2);
    t1 = vextq_f32(t1, t1, 2); // swap the higher half and the lower half
    float32x4_t t2 = vextq_f32(vec1, vec3, 2);
    float32x4_t t3 = vextq_f32(vec3, vec2, 2);
    t3 = vextq_f32(t3, t3, 2);
    
    vec1 = t1;
    vec2 = t2;
    vec3 = t3;
}

// just like swizzle, but the low part and the high part are swapped
inline void swizzle2(float32x4_t& vec1, float32x4_t& vec2, float32x4_t& vec3)
{
    // original:
    // vec1: 00re 00img 01re 01img
    // vec2: 02re 02img 10re 10img
    // vec3: 11re 11img 12re 12img

    // aim:
    // vec1: 10re 10img 00re 00img
    // vec2: 11re 11img 01re 01img
    // vec3: 12re 12img 02re 02img
    float32x4_t t1 = vextq_f32(vec2, vec1, 2);
    float32x4_t t3 = vextq_f32(vec3, vec2, 2);
    float32x4_t t2 = vextq_f32(vec1, vec3, 2);
    t2 = vextq_f32(t2, t2, 2);
    
    vec1 = t1;
    vec2 = t2;
    vec3 = t3;
}

// the inverse operation of swizzle
inline void deswizzle(float32x4_t& v1, float32x4_t& v2, float32x4_t& v3)
{
    float32x4_t t1 = vextq_f32(v1, v1, 2);
    float32x4_t t3 = vextq_f32(v3, v3, 2);

    v1 = vextq_f32(t1, v2, 2);
    v3 = vextq_f32(v2, t3, 2);
    v2 = vextq_f32(t3, t1, 2);
}

// can be inline function or macro
inline void change_sign(float32x4_t& v1, float32x4_t& v2, float32x4_t& v3,
                        uint32x4_t signs)
{
    v1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v1), signs));
    v2 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v2), signs));
    v3 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v3), signs));
}

// adj(3x3 color matrix) * halfspinor
// halfspinor in hs1 hs2 hs3
// result is also in hs1 hs2 hs3
void mat_hvv(float32x4_t& hs1, float32x4_t& hs2, float32x4_t& hs3,
             GaugeMat mat)
{
    static uint32_t signs24UInt[4] __attribute__((aligned(16))) = {0x0, 0x80000000, 0x0, 0x80000000};
    uint32x4_t signs24 = vld1q_u32(signs24UInt);

    float32x4_t v1 = hs1;
    float32x4_t v2 = hs2;
    float32x4_t v3 = hs3;

    float32x4_t m1 = vld1q_dup_f32((float*)&mat[0][0][0]);
    float32x4_t m2 = vld1q_dup_f32((float*)&mat[1][0][0]);
    float32x4_t m3 = vld1q_dup_f32((float*)&mat[2][0][0]);

    float32x4_t acc1 = vmulq_f32(m1, v1);
    float32x4_t acc2 = vmulq_f32(m2, v1);
    float32x4_t acc3 = vmulq_f32(m3, v1);

    m1 = vld1q_dup_f32((float*)&mat[0][1][0]);
    m2 = vld1q_dup_f32((float*)&mat[1][1][0]);
    m3 = vld1q_dup_f32((float*)&mat[2][1][0]);

    acc1 = vfmaq_f32(acc1, m1, v2);
    acc2 = vfmaq_f32(acc2, m2, v2);
    acc3 = vfmaq_f32(acc3, m3, v2);

    m1 = vld1q_dup_f32((float*)&mat[0][2][0]);
    m2 = vld1q_dup_f32((float*)&mat[1][2][0]);
    m3 = vld1q_dup_f32((float*)&mat[2][2][0]);

    acc1 = vfmaq_f32(acc1, m1, v3);
    acc2 = vfmaq_f32(acc2, m2, v3);
    acc3 = vfmaq_f32(acc3, m3, v3);

    // adj means conjugate, a+bi => a-bi
    reverse_real_img(v1, v2, v3);
    change_sign(v1, v2, v3, signs24);

    m1 = vld1q_dup_f32((float*)&mat[0][0][1]);
    m2 = vld1q_dup_f32((float*)&mat[1][0][1]);
    m3 = vld1q_dup_f32((float*)&mat[2][0][1]);

    acc1 = vfmaq_f32(acc1, m1, v1);
    acc2 = vfmaq_f32(acc2, m2, v1);
    acc3 = vfmaq_f32(acc3, m3, v1);

    m1 = vld1q_dup_f32((float*)&mat[0][1][1]);
    m2 = vld1q_dup_f32((float*)&mat[1][1][1]);
    m3 = vld1q_dup_f32((float*)&mat[2][1][1]);

    acc1 = vfmaq_f32(acc1, m1, v2);
    acc2 = vfmaq_f32(acc2, m2, v2);
    acc3 = vfmaq_f32(acc3, m3, v2);

    m1 = vld1q_dup_f32((float*)&mat[0][2][1]);
    m2 = vld1q_dup_f32((float*)&mat[1][2][1]);
    m3 = vld1q_dup_f32((float*)&mat[2][2][1]);
    acc1 = vfmaq_f32(acc1, m1, v3);
    acc2 = vfmaq_f32(acc2, m2, v3);
    acc3 = vfmaq_f32(acc3, m3, v3);

    // done
    hs1 = acc1;
    hs2 = acc2;
    hs3 = acc3;
}

// 3x3 color matrix * halfspinor
// halfspinor in hs1 hs2 hs3
// result is also in hs1 hs2 hs3
void mat_mvv(float32x4_t& hs1, float32x4_t& hs2, float32x4_t& hs3,
             GaugeMat mat)
{
    static uint32_t signs13UInt[] = {0x80000000, 0x0, 0x80000000, 0x0};
    uint32x4_t signs13 = vld1q_u32(signs13UInt);
    
    float32x4_t m1, m2, m3;
    m1 = vld1q_dup_f32((float*)&mat[0][0][0]);
    m2 = vld1q_dup_f32((float*)&mat[0][1][0]);
    m3 = vld1q_dup_f32((float*)&mat[0][2][0]);

    float32x4_t acc1, acc2, acc3;
    acc1 = vmulq_f32(m1, hs1);
    acc2 = vmulq_f32(m2, hs1);
    acc3 = vmulq_f32(m3, hs1);

    m1 = vld1q_dup_f32((float*)&mat[1][0][0]);
    m2 = vld1q_dup_f32((float*)&mat[1][1][0]);
    m3 = vld1q_dup_f32((float*)&mat[1][2][0]);

    acc1 = vfmaq_f32(acc1, m1, hs2);
    acc2 = vfmaq_f32(acc2, m2, hs2);
    acc3 = vfmaq_f32(acc3, m3, hs2);

    m1 = vld1q_dup_f32((float*)&mat[2][0][0]);
    m2 = vld1q_dup_f32((float*)&mat[2][1][0]);
    m3 = vld1q_dup_f32((float*)&mat[2][2][0]);

    acc1 = vfmaq_f32(acc1, m1, hs3);
    acc2 = vfmaq_f32(acc2, m2, hs3);
    acc3 = vfmaq_f32(acc3, m3, hs3);

    reverse_real_img(hs1, hs2, hs3);
    change_sign(hs1, hs2, hs3, signs13);

    m1 = vld1q_dup_f32((float*)&mat[0][0][1]);
    m2 = vld1q_dup_f32((float*)&mat[0][1][1]);
    m3 = vld1q_dup_f32((float*)&mat[0][2][1]);

    acc1 = vfmaq_f32(acc1, m1, hs1);
    acc2 = vfmaq_f32(acc2, m2, hs1);
    acc3 = vfmaq_f32(acc3, m3, hs1);

    m1 = vld1q_dup_f32((float*)&mat[1][0][1]);
    m2 = vld1q_dup_f32((float*)&mat[1][1][1]);
    m3 = vld1q_dup_f32((float*)&mat[1][2][1]);

    acc1 = vfmaq_f32(acc1, m1, hs2);
    acc2 = vfmaq_f32(acc2, m2, hs2);
    acc3 = vfmaq_f32(acc3, m3, hs2);

    m1 = vld1q_dup_f32((float*)&mat[2][0][1]);
    m2 = vld1q_dup_f32((float*)&mat[2][1][1]);
    m3 = vld1q_dup_f32((float*)&mat[2][2][1]);

    acc1 = vfmaq_f32(acc1, m1, hs3);
    acc2 = vfmaq_f32(acc2, m2, hs3);
    acc3 = vfmaq_f32(acc3, m3, hs3);

    // done
    hs1 = acc1;
    hs2 = acc2;
    hs3 = acc3;
}

// (a0-i*a3, a1-i*a2)
void decomp_gamma0_minus_impl(Spinor src,
                              float32x4_t& res1, float32x4_t& res2,
                              float32x4_t& res3)
{
    static uint32_t signs24UInt[4] __attribute__((aligned(16))) = {0x00000000, 0x80000000, 0x00000000, 0x80000000};
    uint32x4_t signs24 = vld1q_u32(signs24UInt);
    
    float32x4_t vec1, vec2, vec3, vec4, vec5, vec6;
    vec1 = vld1q_f32((float*)&src[0][0][0]);
    vec2 = vld1q_f32((float*)&src[0][2][0]);
    vec3 = vld1q_f32((float*)&src[1][1][0]);
    vec4 = vld1q_f32((float*)&src[2][0][0]);
    vec5 = vld1q_f32((float*)&src[2][2][0]);
    vec6 = vld1q_f32((float*)&src[3][1][0]);

    swizzle(vec1, vec2, vec3);
    swizzle2(vec4, vec5, vec6);

    // * i ==> reverse_real_img
    reverse_real_img(vec4, vec5, vec6);
    
    // make the second lane and fourth lane negative
    change_sign(vec4, vec5, vec6, signs24);

    // result in res1 res2 res3
    res1 = vaddq_f32(vec1, vec4);
    res2 = vaddq_f32(vec2, vec5);
    res3 = vaddq_f32(vec3, vec6);
}

// (a0+a3, a1-a2)
void decomp_gamma1_minus_impl(Spinor src,
                              float32x4_t& res1, float32x4_t& res2,
                              float32x4_t& res3)
{
    static uint32_t signs34UInt[4] __attribute__((aligned(16))) = {0x0, 0x0, 0x80000000, 0x80000000};
    uint32x4_t signs34 = vld1q_u32(signs34UInt);
    
    float32x4_t vec1, vec2, vec3, vec4, vec5, vec6;
    vec1 = vld1q_f32((float*)&src[0][0][0]);
    vec2 = vld1q_f32((float*)&src[0][2][0]);
    vec3 = vld1q_f32((float*)&src[1][1][0]);
    vec4 = vld1q_f32((float*)&src[2][0][0]);
    vec5 = vld1q_f32((float*)&src[2][2][0]);
    vec6 = vld1q_f32((float*)&src[3][1][0]);

    swizzle(vec1, vec2, vec3);
    swizzle2(vec4, vec5, vec6);

    change_sign(vec4, vec5, vec6, signs34);

    res1 = vaddq_f32(vec1, vec4);
    res2 = vaddq_f32(vec2, vec5);
    res3 = vaddq_f32(vec3, vec6);
}

// (a0-i*a2, a1+i*a3)
void decomp_gamma2_minus_impl(Spinor src,
                              float32x4_t& res1, float32x4_t& res2,
                              float32x4_t& res3)
{
    static uint32_t signs23UInt[4] __attribute__((aligned(16))) = {0x0, 0x80000000, 0x80000000, 0x0};
    uint32x4_t signs23 = vld1q_u32(signs23UInt);
    
    float32x4_t vec1, vec2, vec3, vec4, vec5, vec6;
    vec1 = vld1q_f32((float*)&src[0][0][0]);
    vec2 = vld1q_f32((float*)&src[0][2][0]);
    vec3 = vld1q_f32((float*)&src[1][1][0]);
    vec4 = vld1q_f32((float*)&src[2][0][0]);
    vec5 = vld1q_f32((float*)&src[2][2][0]);
    vec6 = vld1q_f32((float*)&src[3][1][0]);

    swizzle(vec1, vec2, vec3);
    swizzle(vec4, vec5, vec6);

    reverse_real_img(vec4, vec5, vec6);

    change_sign(vec4, vec5, vec6, signs23);

    res1 = vaddq_f32(vec1, vec4);
    res2 = vaddq_f32(vec2, vec5);
    res3 = vaddq_f32(vec3, vec6);
}

// (a0-a2, a1-a3)
void decomp_gamma3_minus_impl(Spinor src,
                              float32x4_t& res1, float32x4_t& res2,
                              float32x4_t& res3)
{   
    float32x4_t vec1, vec2, vec3, vec4, vec5, vec6;
    vec1 = vld1q_f32((float*)&src[0][0][0]);
    vec2 = vld1q_f32((float*)&src[0][2][0]);
    vec3 = vld1q_f32((float*)&src[1][1][0]);
    vec4 = vld1q_f32((float*)&src[2][0][0]);
    vec5 = vld1q_f32((float*)&src[2][2][0]);
    vec6 = vld1q_f32((float*)&src[3][1][0]);

    swizzle(vec1, vec2, vec3);
    swizzle(vec4, vec5, vec6);

    // no need to change signs. do subtract for all
    res1 = vsubq_f32(vec1, vec4);
    res2 = vsubq_f32(vec2, vec5);
    res3 = vsubq_f32(vec3, vec6);
}

// (a0+i*a3, a1+i*a2)
void decomp_gamma0_plus_impl(Spinor src,
                             float32x4_t& res1, float32x4_t& res2,
                             float32x4_t& res3)
{
    static uint32_t signs13UInt[4] __attribute__((aligned(16))) = {0x80000000, 0x0, 0x80000000, 0x0};
    uint32x4_t signs13 = vld1q_u32(signs13UInt);
    
    float32x4_t vec1, vec2, vec3, vec4, vec5, vec6;
    vec1 = vld1q_f32((float*)&src[0][0][0]);
    vec2 = vld1q_f32((float*)&src[0][2][0]);
    vec3 = vld1q_f32((float*)&src[1][1][0]);
    vec4 = vld1q_f32((float*)&src[2][0][0]);
    vec5 = vld1q_f32((float*)&src[2][2][0]);
    vec6 = vld1q_f32((float*)&src[3][1][0]);

    swizzle(vec1, vec2, vec3);
    swizzle2(vec4, vec5, vec6);

    reverse_real_img(vec4, vec5, vec6);

    change_sign(vec4, vec5, vec6, signs13);

    res1 = vaddq_f32(vec1, vec4);
    res2 = vaddq_f32(vec2, vec5);
    res3 = vaddq_f32(vec3, vec6);
}

// (a0-a3, a1+a2)
void decomp_gamma1_plus_impl(Spinor src, 
                             float32x4_t& res1, float32x4_t& res2,
                             float32x4_t& res3)
{
    static uint32_t signs12UInt[4] = {0x80000000, 0x80000000, 0x0, 0x0};
    uint32x4_t signs12 = vld1q_u32(signs12UInt);

    float32x4_t vec1, vec2, vec3, vec4, vec5, vec6;
    vec1 = vld1q_f32((float*)&src[0][0][0]);
    vec2 = vld1q_f32((float*)&src[0][2][0]);
    vec3 = vld1q_f32((float*)&src[1][1][0]);
    vec4 = vld1q_f32((float*)&src[2][0][0]);
    vec5 = vld1q_f32((float*)&src[2][2][0]);
    vec6 = vld1q_f32((float*)&src[3][1][0]);

    swizzle(vec1, vec2, vec3);
    swizzle2(vec4, vec5, vec6);

    change_sign(vec4, vec5, vec6, signs12);

    res1 = vaddq_f32(vec1, vec4);
    res2 = vaddq_f32(vec2, vec5);
    res3 = vaddq_f32(vec3, vec6);
}

// (a0+i*a2, a1-i*a3)
void decomp_gamma2_plus_impl(Spinor src, 
                             float32x4_t& res1, float32x4_t& res2,
                             float32x4_t& res3)
{
    static uint32_t signs14UInt[4] = {0x80000000, 0x0, 0x0, 0x80000000};
    uint32x4_t signs14 = vld1q_u32(signs14UInt);

    float32x4_t vec1, vec2, vec3, vec4, vec5, vec6;
    vec1 = vld1q_f32((float*)&src[0][0][0]);
    vec2 = vld1q_f32((float*)&src[0][2][0]);
    vec3 = vld1q_f32((float*)&src[1][1][0]);
    vec4 = vld1q_f32((float*)&src[2][0][0]);
    vec5 = vld1q_f32((float*)&src[2][2][0]);
    vec6 = vld1q_f32((float*)&src[3][1][0]);

    swizzle(vec1, vec2, vec3);
    swizzle(vec4, vec5, vec6);

    reverse_real_img(vec4, vec5, vec6);

    change_sign(vec4, vec5, vec6, signs14);

    res1 = vaddq_f32(vec1, vec4);
    res2 = vaddq_f32(vec2, vec5);
    res3 = vaddq_f32(vec3, vec6);
}

// (a0+a2, a1+a3)
void decomp_gamma3_plus_impl(Spinor src, 
                             float32x4_t& res1, float32x4_t& res2,
                             float32x4_t& res3)
{
    float32x4_t vec1, vec2, vec3, vec4, vec5, vec6;
    vec1 = vld1q_f32((float*)&src[0][0][0]);
    vec2 = vld1q_f32((float*)&src[0][2][0]);
    vec3 = vld1q_f32((float*)&src[1][1][0]);
    vec4 = vld1q_f32((float*)&src[2][0][0]);
    vec5 = vld1q_f32((float*)&src[2][2][0]);
    vec6 = vld1q_f32((float*)&src[3][1][0]);

    swizzle(vec1, vec2, vec3);
    swizzle(vec4, vec5, vec6);

    res1 = vaddq_f32(vec1, vec4);
    res2 = vaddq_f32(vec2, vec5);
    res3 = vaddq_f32(vec3, vec6);
}

// spinprojdirminus
inline void decomp_gamma0_minus(Spinor src, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma0_minus_impl(src, vec1, vec2, vec3);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_gamma1_minus(Spinor src, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma1_minus_impl(src, vec1, vec2, vec3);
    
    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_gamma2_minus(Spinor src, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma2_minus_impl(src, vec1, vec2, vec3);
    
    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_gamma3_minus(Spinor src, HalfSpinor dst)
{   
    float32x4_t vec1, vec2, vec3;
    decomp_gamma3_minus_impl(src, vec1, vec2, vec3);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

// spinprojdirplus
inline void decomp_gamma0_plus(Spinor src, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma0_plus_impl(src, vec1, vec2, vec3);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_gamma1_plus(Spinor src, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma1_plus_impl(src, vec1, vec2, vec3);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_gamma2_plus(Spinor src, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma2_plus_impl(src, vec1, vec2, vec3);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_gamma3_plus(Spinor src, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma3_plus_impl(src, vec1, vec2, vec3);    

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

// adj(u) * spinprojdirplus
inline void decomp_hvv_gamma0_plus(Spinor src, GaugeMat mat, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma0_plus_impl(src, vec1, vec2, vec3);
    // now the half spinor is in [vec1 vec2 vec3]
    // begin mat hvv
    mat_hvv(vec1, vec2, vec3, mat);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_hvv_gamma1_plus(Spinor src, GaugeMat mat, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma1_plus_impl(src, vec1, vec2, vec3);
    
    mat_hvv(vec1, vec2, vec3, mat);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_hvv_gamma2_plus(Spinor src, GaugeMat mat, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma2_plus_impl(src, vec1, vec2, vec3);
    mat_hvv(vec1, vec2, vec3, mat);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_hvv_gamma3_plus(Spinor src, GaugeMat mat, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma3_plus_impl(src, vec1, vec2, vec3);    
    mat_hvv(vec1, vec2, vec3, mat);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}


// adj(u) * spinprojdirminus
inline void decomp_hvv_gamma0_minus(Spinor src, GaugeMat mat, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma0_minus_impl(src, vec1, vec2, vec3);
    mat_hvv(vec1, vec2, vec3, mat);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_hvv_gamma1_minus(Spinor src, GaugeMat mat, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma1_minus_impl(src, vec1, vec2, vec3);
    mat_hvv(vec1, vec2, vec3, mat);
    
    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_hvv_gamma2_minus(Spinor src, GaugeMat mat, HalfSpinor dst)
{
    float32x4_t vec1, vec2, vec3;
    decomp_gamma2_minus_impl(src, vec1, vec2, vec3);
    mat_hvv(vec1, vec2, vec3, mat);
    
    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

inline void decomp_hvv_gamma3_minus(Spinor src, GaugeMat mat, HalfSpinor dst)
{   
    float32x4_t vec1, vec2, vec3;
    decomp_gamma3_minus_impl(src, vec1, vec2, vec3);
    mat_hvv(vec1, vec2, vec3, mat);

    vst1q_f32((float*)&dst[0][0][0], vec1);
    vst1q_f32((float*)&dst[1][0][0], vec2);
    vst1q_f32((float*)&dst[2][0][0], vec3);
}

void mvv_recons_4dir_minus(HalfSpinor src1, HalfSpinor src2, HalfSpinor src3, HalfSpinor src4,
                           GaugeMat mat1, GaugeMat mat2, GaugeMat mat3, GaugeMat mat4,
                           Spinor dst)
{
    static uint32_t signs13UInt[4] __attribute__((aligned(16))) = {0x80000000, 0x0, 0x80000000, 0x0};
    static uint32_t signs12UInt[4] __attribute__((aligned(16))) = {0x80000000, 0x80000000, 0x0, 0x0};
    static uint32_t signs14UInt[4] __attribute__((aligned(16))) = {0x80000000, 0x0, 0x0, 0x80000000};
    
    
    float32x4_t upperSum[3];
    float32x4_t lowerSum[3];

    // dir0 (a0 a1) -> (a0 a1 i*a1 i*a0)
    {
        uint32x4_t signs13 = vld1q_u32(signs13UInt);
        
        float32x4_t v1 = vld1q_f32((float*)&src1[0][0][0]);
        float32x4_t v2 = vld1q_f32((float*)&src1[1][0][0]);
        float32x4_t v3 = vld1q_f32((float*)&src1[2][0][0]);

        mat_mvv(v1, v2, v3, mat1); // result in v1 v2 v3    
        // save upper half (a0 a1)
        upperSum[0] = v1;
        upperSum[1] = v2;
        upperSum[2] = v3;
        // do reconstruct
        reverse_elements(v1, v2, v3);
        change_sign(v1, v2, v3, signs13);
        
        lowerSum[0] = v1;
        lowerSum[1] = v2;
        lowerSum[2] = v3;
    }

    // dir1 (a0 a1) -> (a0 a1 -a1 a0)
    {
        uint32x4_t signs12 = vld1q_u32(signs12UInt);
        
        float32x4_t v1 = vld1q_f32((float*)&src2[0][0][0]);
        float32x4_t v2 = vld1q_f32((float*)&src2[1][0][0]);
        float32x4_t v3 = vld1q_f32((float*)&src2[2][0][0]);
        
        mat_mvv(v1, v2, v3, mat2);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);
        
        v1 = vextq_f32(v1, v1, 2);
        v2 = vextq_f32(v2, v2, 2);
        v3 = vextq_f32(v3, v3, 2);

        change_sign(v1, v2, v3, signs12);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // dir2 (a0 a1) -> (a0 a1 i*a0 -i*a1)
    {
        uint32x4_t signs14 = vld1q_u32(signs14UInt);
        float32x4_t v1 = vld1q_f32((float*)&src3[0][0][0]);
        float32x4_t v2 = vld1q_f32((float*)&src3[1][0][0]);
        float32x4_t v3 = vld1q_f32((float*)&src3[2][0][0]);
        
        mat_mvv(v1, v2, v3, mat3);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        reverse_real_img(v1, v2, v3);
        change_sign(v1, v2, v3, signs14);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // dir3 (a0 a1) -> (a0 a1 -a0 -a1)
    {
        float32x4_t v1 = vld1q_f32((float*)&src4[0][0][0]);
        float32x4_t v2 = vld1q_f32((float*)&src4[1][0][0]);
        float32x4_t v3 = vld1q_f32((float*)&src4[2][0][0]);
        
        mat_mvv(v1, v2, v3, mat4);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        lowerSum[0] = vsubq_f32(lowerSum[0], v1);
        lowerSum[1] = vsubq_f32(lowerSum[1], v2);
        lowerSum[2] = vsubq_f32(lowerSum[2], v3);
    }

    // finally do stores
    // don't do desizzling because it's partial sum 
    vst1q_f32((float*)&dst[0][0][0], upperSum[0]);
    vst1q_f32((float*)&dst[0][2][0], upperSum[1]);
    vst1q_f32((float*)&dst[1][1][0], upperSum[2]);
    vst1q_f32((float*)&dst[2][0][0], lowerSum[0]);
    vst1q_f32((float*)&dst[2][2][0], lowerSum[1]);
    vst1q_f32((float*)&dst[3][1][0], lowerSum[2]);
}

void mvv_recons_4dir_plus(HalfSpinor src1, HalfSpinor src2, HalfSpinor src3, HalfSpinor src4,
                          GaugeMat mat1, GaugeMat mat2, GaugeMat mat3, GaugeMat mat4,
                          Spinor dst)
{
    uint32_t signs24UInt[4] __attribute__((aligned(16))) = {0, 0x80000000, 0, 0x80000000};
    uint32_t signs34UInt[4] __attribute__((aligned(16))) = {0, 0, 0x80000000, 0x80000000};
    uint32_t signs23UInt[4] __attribute__((aligned(16))) = {0, 0x80000000, 0x80000000, 0};

    uint32x4_t signs24 = vld1q_u32(signs24UInt);
    uint32x4_t signs34 = vld1q_u32(signs34UInt);
    uint32x4_t signs23 = vld1q_u32(signs23UInt);
    
    float32x4_t upperSum[3];
    float32x4_t lowerSum[3];

    // dir0 (a0 a1) -> (a0 a1 -i*a1 -i*a0)
    {
        float32x4_t v1, v2, v3;
        v1 = vld1q_f32((float*)&src1[0][0][0]);
        v2 = vld1q_f32((float*)&src1[1][0][0]);
        v3 = vld1q_f32((float*)&src1[2][0][0]);
        mat_mvv(v1, v2, v3, mat1);

        upperSum[0] = v1;
        upperSum[1] = v2;
        upperSum[2] = v3;

        reverse_elements(v1, v2, v3);
        change_sign(v1, v2, v3, signs24);

        lowerSum[0] = v1;
        lowerSum[1] = v2;
        lowerSum[2] = v3;
    }

    // dir1 (a0 a1) -> (a0 a1 a1 -a0)
    {
        auto v1 = vld1q_f32((float*)&src2[0][0][0]);
        auto v2 = vld1q_f32((float*)&src2[1][0][0]);
        auto v3 = vld1q_f32((float*)&src2[2][0][0]);
        mat_mvv(v1, v2, v3, mat2);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        v1 = vextq_f32(v1, v1, 2);
        v2 = vextq_f32(v2, v2, 2);
        v3 = vextq_f32(v3, v3, 2);
        change_sign(v1, v2, v3, signs34);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // dir2 (a0 a1) -> (a0 a1 -i*a0 i*a1)
    {
        auto v1 = vld1q_f32((float*)&src3[0][0][0]);
        auto v2 = vld1q_f32((float*)&src3[1][0][0]);
        auto v3 = vld1q_f32((float*)&src3[2][0][0]);
        mat_mvv(v1, v2, v3, mat3);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        reverse_real_img(v1, v2, v3);
        change_sign(v1, v2, v3, signs23);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // dir3 (a0 a1) -> (a0 a1 a0 a1)
    {
        auto v1 = vld1q_f32((float*)&src4[0][0][0]);
        auto v2 = vld1q_f32((float*)&src4[1][0][0]);
        auto v3 = vld1q_f32((float*)&src4[2][0][0]);
        mat_mvv(v1, v2, v3, mat4);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // done. store
    vst1q_f32((float*)&dst[0][0][0], upperSum[0]);
    vst1q_f32((float*)&dst[0][2][0], upperSum[1]);
    vst1q_f32((float*)&dst[1][1][0], upperSum[2]);
    vst1q_f32((float*)&dst[2][0][0], lowerSum[0]);
    vst1q_f32((float*)&dst[2][2][0], lowerSum[1]);
    vst1q_f32((float*)&dst[3][1][0], lowerSum[2]);
}

void recons_4dir_plus(HalfSpinor src1, HalfSpinor src2,
                      HalfSpinor src3, HalfSpinor src4,
                      Spinor dst)
{
    uint32_t signs24UInt[4] __attribute__((aligned(16))) = {0, 0x80000000, 0, 0x80000000};
    uint32_t signs34UInt[4] __attribute__((aligned(16))) = {0, 0, 0x80000000, 0x80000000};
    uint32_t signs23UInt[4] __attribute__((aligned(16))) = {0, 0x80000000, 0x80000000, 0};

    uint32x4_t signs24 = vld1q_u32(signs24UInt);
    uint32x4_t signs34 = vld1q_u32(signs34UInt);
    uint32x4_t signs23 = vld1q_u32(signs23UInt);
    
    float32x4_t upperSum[3];
    float32x4_t lowerSum[3];

    // read partial sum
    upperSum[0] = vld1q_f32((float*)&dst[0][0][0]);
    upperSum[1] = vld1q_f32((float*)&dst[0][2][0]);
    upperSum[2] = vld1q_f32((float*)&dst[1][1][0]);
    lowerSum[0] = vld1q_f32((float*)&dst[2][0][0]);
    lowerSum[1] = vld1q_f32((float*)&dst[2][2][0]);
    lowerSum[2] = vld1q_f32((float*)&dst[3][1][0]);

    // dir0 (a0 a1) -> (a0 a1 -i*a1 -i*a0)
    {
        auto v1 = vld1q_f32((float*)&src1[0][0][0]);
        auto v2 = vld1q_f32((float*)&src1[1][0][0]);
        auto v3 = vld1q_f32((float*)&src1[2][0][0]);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        reverse_elements(v1, v2, v3);
        change_sign(v1, v2, v3, signs24);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // dir1 (a0 a1) -> (a0 a1 a1 -a0)
    {
        auto v1 = vld1q_f32((float*)&src2[0][0][0]);
        auto v2 = vld1q_f32((float*)&src2[1][0][0]);
        auto v3 = vld1q_f32((float*)&src2[2][0][0]);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        v1 = vextq_f32(v1, v1, 2);
        v2 = vextq_f32(v2, v2, 2);
        v3 = vextq_f32(v3, v3, 2);
        change_sign(v1, v2, v3, signs34);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // dir2 (a0 a1) -> (a0 a1 -i*a0 -i*a1)
    {
        auto v1 = vld1q_f32((float*)&src3[0][0][0]);
        auto v2 = vld1q_f32((float*)&src3[1][0][0]);
        auto v3 = vld1q_f32((float*)&src3[2][0][0]);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        reverse_real_img(v1, v2, v3);
        change_sign(v1, v2, v3, signs23);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // dir3 (a0 a1) -> (a0 a1 a0 a1)
    {
        auto v1 = vld1q_f32((float*)&src4[0][0][0]);
        auto v2 = vld1q_f32((float*)&src4[1][0][0]);
        auto v3 = vld1q_f32((float*)&src4[2][0][0]);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // done. deswizzle and store
    deswizzle(upperSum[0], upperSum[1], upperSum[2]);
    deswizzle(lowerSum[0], lowerSum[1], lowerSum[2]);
    
    vst1q_f32((float*)&dst[0][0][0], upperSum[0]);
    vst1q_f32((float*)&dst[0][2][0], upperSum[1]);
    vst1q_f32((float*)&dst[1][1][0], upperSum[2]);
    vst1q_f32((float*)&dst[2][0][0], lowerSum[0]);
    vst1q_f32((float*)&dst[2][2][0], lowerSum[1]);
    vst1q_f32((float*)&dst[3][1][0], lowerSum[2]);
}

void recons_4dir_minus(HalfSpinor src1, HalfSpinor src2,
                       HalfSpinor src3, HalfSpinor src4, 
                       Spinor dst)
{
    static uint32_t signs13UInt[4] __attribute__((aligned(16))) = {0x80000000, 0x0, 0x80000000, 0x0};
    static uint32_t signs12UInt[4] __attribute__((aligned(16))) = {0x80000000, 0x80000000, 0x0, 0x0};
    static uint32_t signs14UInt[4] __attribute__((aligned(16))) = {0x80000000, 0x0, 0x0, 0x80000000};
        
    float32x4_t upperSum[3];
    float32x4_t lowerSum[3];
    upperSum[0] = vld1q_f32((float*)&dst[0][0][0]);
    upperSum[1] = vld1q_f32((float*)&dst[0][2][0]);
    upperSum[2] = vld1q_f32((float*)&dst[1][1][0]);
    lowerSum[0] = vld1q_f32((float*)&dst[2][0][0]);
    lowerSum[1] = vld1q_f32((float*)&dst[2][2][0]);
    lowerSum[2] = vld1q_f32((float*)&dst[3][1][0]);

    // dir0 (a0 a1) -> (a0 a1 i*a1 i*a0)
    {
        uint32x4_t signs13 = vld1q_u32(signs13UInt);        
        float32x4_t v1 = vld1q_f32((float*)&src1[0][0][0]);
        float32x4_t v2 = vld1q_f32((float*)&src1[1][0][0]);
        float32x4_t v3 = vld1q_f32((float*)&src1[2][0][0]);

        // save upper half (a0 a1)
        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);
        // do reconstruct
        reverse_elements(v1, v2, v3);
        change_sign(v1, v2, v3, signs13);
        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // dir1 (a0 a1) -> (a0 a1 -a1 -a0)
    {
        uint32x4_t signs12 = vld1q_u32(signs12UInt);
        
        auto v1 = vld1q_f32((float*)&src2[0][0][0]);
        auto v2 = vld1q_f32((float*)&src2[1][0][0]);
        auto v3 = vld1q_f32((float*)&src2[2][0][0]);        

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);
        
        v1 = vextq_f32(v1, v1, 2);
        v2 = vextq_f32(v2, v2, 2);
        v3 = vextq_f32(v3, v3, 2);

        change_sign(v1, v2, v3, signs12);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // dir2 (a0 a1) -> (a0 a1 i*a0 -i*a1)
    {
        uint32x4_t signs14 = vld1q_u32(signs14UInt);
        auto v1 = vld1q_f32((float*)&src3[0][0][0]);
        auto v2 = vld1q_f32((float*)&src3[1][0][0]);
        auto v3 = vld1q_f32((float*)&src3[2][0][0]);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        reverse_real_img(v1, v2, v3);
        change_sign(v1, v2, v3, signs14);

        lowerSum[0] = vaddq_f32(lowerSum[0], v1);
        lowerSum[1] = vaddq_f32(lowerSum[1], v2);
        lowerSum[2] = vaddq_f32(lowerSum[2], v3);
    }

    // dir3 (a0 a1) -> (a0 a1 -a0 -a1)
    {
        auto v1 = vld1q_f32((float*)&src4[0][0][0]);
        auto v2 = vld1q_f32((float*)&src4[1][0][0]);
        auto v3 = vld1q_f32((float*)&src4[2][0][0]);

        upperSum[0] = vaddq_f32(upperSum[0], v1);
        upperSum[1] = vaddq_f32(upperSum[1], v2);
        upperSum[2] = vaddq_f32(upperSum[2], v3);

        lowerSum[0] = vsubq_f32(lowerSum[0], v1);
        lowerSum[1] = vsubq_f32(lowerSum[1], v2);
        lowerSum[2] = vsubq_f32(lowerSum[2], v3);
    }

    deswizzle(upperSum[0], upperSum[1], upperSum[2]);
    deswizzle(lowerSum[0], lowerSum[1], lowerSum[2]);
    
    vst1q_f32((float*)&dst[0][0][0], upperSum[0]);
    vst1q_f32((float*)&dst[0][2][0], upperSum[1]);
    vst1q_f32((float*)&dst[1][1][0], upperSum[2]);
    vst1q_f32((float*)&dst[2][0][0], lowerSum[0]);
    vst1q_f32((float*)&dst[2][2][0], lowerSum[1]);
    vst1q_f32((float*)&dst[3][1][0], lowerSum[2]);
}

} // namespace anonymous

#endif
