// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Konstantinos Margaritis <markos@codex.gr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_ALTIVEC_H
#define EIGEN_PACKET_MATH_ALTIVEC_H

namespace Eigen {

namespace internal {

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 4
#endif

#ifndef EIGEN_HAS_FUSE_CJMADD
#define EIGEN_HAS_FUSE_CJMADD 1
#endif

// NOTE Altivec has 32 registers, but Eigen only accepts a value of 8 or 16
#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 16
#endif

typedef __vector4 Packet;

__declspec(passinreg) struct Packet4f : __vector4
{
	EIGEN_STRONG_INLINE Packet4f() {}
    EIGEN_STRONG_INLINE Packet4f(__vector4 v_): __vector4(v_) {}
};

__declspec(passinreg) struct Packet4i : __vector4
{
	EIGEN_STRONG_INLINE Packet4i() {}
    EIGEN_STRONG_INLINE Packet4i(__vector4 v_): __vector4(v_) {}
};

struct Packet4ui : __vector4
{
	EIGEN_STRONG_INLINE Packet4ui() {}
    EIGEN_STRONG_INLINE Packet4ui(__vector4 v_): __vector4(v_) {}
};

struct EIGEN_ALIGN16 Packet4fInit
{
  union
  {
    float f[4];
    __vector4 v;
  };

  EIGEN_STRONG_INLINE operator __vector4() const { return v; }
  EIGEN_STRONG_INLINE operator const float*() const { return f; }
};

struct EIGEN_ALIGN16 Packet4iInit
{
  union
  {
    int i[4];
    __vector4 v;
  };

  EIGEN_STRONG_INLINE operator __vector4() const { return v; }
  EIGEN_STRONG_INLINE operator const int*() const { return i; }
};

// We don't want to write the same code all the time, but we need to reuse the constants
// and it doesn't really work to declare them global, so we define macros instead

#define _EIGEN_DECLARE_CONST_FAST_Packet4f(NAME,X) \
  Packet4f p4f_##NAME = (Packet4f) __vspltisw(X)

#define _EIGEN_DECLARE_CONST_FAST_Packet4i(NAME,X) \
  Packet4i p4i_##NAME = (Packet4i) __vspltisw(X)

#define _EIGEN_DECLARE_CONST_Packet4f(NAME,X) \
  Packet4f p4f_##NAME = pset1<Packet4f>(X)

#define _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(NAME,X) \
  Packet4f p4f_##NAME = (Packet4f)vreinterpretq_f32_u32(pset1<int>(X))

#define _EIGEN_DECLARE_CONST_Packet4i(NAME,X) \
  Packet4i p4i_##NAME = pset1<Packet4i>(X)

// Define global static constants:
static Packet4fInit p4f_COUNTDOWN = { 3.0, 2.0, 1.0, 0.0 };
static Packet4iInit p4i_COUNTDOWN = { 3, 2, 1, 0 };
static Packet4iInit p4i_SEL_STEP1 = { 0, 1, 0, 1 };
static Packet4iInit p4i_SEL_STEP2 = { 0, 0, 1, 1 };

// Permutations
#define _EIGEN_PERMUTATION_MASK(x,y,z,w) (((x)<<6)|((y)<<4)|((z)<<2)|w)
const unsigned int pic_FORWARD = _EIGEN_PERMUTATION_MASK(0,1,2,3);
const unsigned int pic_REVERSE = _EIGEN_PERMUTATION_MASK(3,2,1,0);
const unsigned int pic_DUPLICATE = _EIGEN_PERMUTATION_MASK(0,0,1,1);
const unsigned int pic_REV = _EIGEN_PERMUTATION_MASK(1,0,3,2);
const unsigned int pic_REV2 = _EIGEN_PERMUTATION_MASK(2,3,0,1);

static _EIGEN_DECLARE_CONST_FAST_Packet4f(ZERO, 0);
static _EIGEN_DECLARE_CONST_FAST_Packet4i(ZERO, 0);
static _EIGEN_DECLARE_CONST_FAST_Packet4i(ONE,1);
static _EIGEN_DECLARE_CONST_FAST_Packet4i(MINUS16,-16);
static _EIGEN_DECLARE_CONST_FAST_Packet4i(MINUS1,-1);
static Packet4f p4f_ONE = __vcfsx(p4i_ONE, 0);
static Packet4f p4f_ZERO_ = (Packet4f) __vsl((Packet4ui)p4i_MINUS1, (Packet4ui)p4i_MINUS1);

template<> struct packet_traits<float>  : default_packet_traits
{
  typedef Packet4f type;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=4,

    // FIXME check the Has*
    HasSin  = 0,
    HasCos  = 0,
    HasLog  = 0,
    HasExp  = 0,
    HasSqrt = 0
  };
};
template<> struct packet_traits<int>    : default_packet_traits
{
  typedef Packet4i type;
  enum {
    // FIXME check the Has*
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=4
  };
};

template<> struct unpacket_traits<Packet4f> { typedef float  type; enum {size=4}; };
template<> struct unpacket_traits<Packet4i> { typedef int    type; enum {size=4}; };
/*
inline std::ostream & operator <<(std::ostream & s, const Packet4f & v)
{
  union {
    Packet4f   v;
    float n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}

inline std::ostream & operator <<(std::ostream & s, const Packet4i & v)
{
  union {
    Packet4i   v;
    int n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}

inline std::ostream & operator <<(std::ostream & s, const Packet4ui & v)
{
  union {
    Packet4ui   v;
    unsigned int n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}

inline std::ostream & operator <<(std::ostream & s, const Packetbi & v)
{
  union {
    Packet4bi v;
    unsigned int n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}
*/
template<> EIGEN_STRONG_INLINE Packet4f pset1<Packet4f>(const float&  from) {
  // Taken from http://developer.apple.com/hardwaredrivers/ve/alignment.html
  float EIGEN_ALIGN16 af[4];
  af[0] = from;
  Packet4f vc = __lvewx (af, 0);
  vc = __vspltw(vc, 0);
  return vc;
}

template<> EIGEN_STRONG_INLINE Packet4i pset1<Packet4i>(const int&    from)   {
  int EIGEN_ALIGN16 ai[4];
  ai[0] = from;
  Packet4i vc = __lvewx(ai, 0);
  vc = __vspltw(vc, 0);
  return vc;
}

template<> EIGEN_STRONG_INLINE Packet4f plset<float>(const float& a) { return __vaddfp(pset1<Packet4f>(a), p4f_COUNTDOWN); }
template<> EIGEN_STRONG_INLINE Packet4i plset<int>(const int& a)     { return __vadduwm(pset1<Packet4i>(a), p4i_COUNTDOWN); }

template<> EIGEN_STRONG_INLINE Packet4f padd<Packet4f>(const Packet4f& a, const Packet4f& b) { return __vaddfp(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i padd<Packet4i>(const Packet4i& a, const Packet4i& b) { return __vadduwm(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f psub<Packet4f>(const Packet4f& a, const Packet4f& b) { return __vsubfp(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i psub<Packet4i>(const Packet4i& a, const Packet4i& b) { return __vsubuwm(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f pnegate(const Packet4f& a) { return __vsubfp(p4f_ZERO, a); }
template<> EIGEN_STRONG_INLINE Packet4i pnegate(const Packet4i& a) { return __vsubuwm(p4i_ZERO, a); }

template<> EIGEN_STRONG_INLINE Packet4f pconj(const Packet4f& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet4i pconj(const Packet4i& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet4f pmul<Packet4f>(const Packet4f& a, const Packet4f& b) { return __vmulfp(a,b); }
/* Commented out: it's actually slower than processing it scalar
 *
template<> EIGEN_STRONG_INLINE Packet4i pmul<Packet4i>(const Packet4i& a, const Packet4i& b)
{
  // Detailed in: http://freevec.org/content/32bit_signed_integer_multiplication_altivec
  //Set up constants, variables
  Packet4i a1, b1, bswap, low_prod, high_prod, prod, prod_, v1sel;

  // Get the absolute values
  a1  = vec_abs(a);
  b1  = vec_abs(b);

  // Get the signs using xor
  Packet4bi sgn = (Packet4bi) vec_cmplt(vec_xor(a, b), p4i_ZERO);

  // Do the multiplication for the asbolute values.
  bswap = (Packet4i) vec_rl((Packet4ui) b1, (Packet4ui) p4i_MINUS16 );
  low_prod = vec_mulo((Packet8i) a1, (Packet8i)b1);
  high_prod = vec_msum((Packet8i) a1, (Packet8i) bswap, p4i_ZERO);
  high_prod = (Packet4i) vec_sl((Packet4ui) high_prod, (Packet4ui) p4i_MINUS16);
  prod = vec_add( low_prod, high_prod );

  // NOR the product and select only the negative elements according to the sign mask
  prod_ = vec_nor(prod, prod);
  prod_ = vec_sel(p4i_ZERO, prod_, sgn);

  // Add 1 to the result to get the negative numbers
  v1sel = vec_sel(p4i_ZERO, p4i_ONE, sgn);
  prod_ = vec_add(prod_, v1sel);

  // Merge the results back to the final vector.
  prod = vec_sel(prod, prod_, sgn);

  return prod;
}
*/
template<> EIGEN_STRONG_INLINE Packet4f pdiv<Packet4f>(const Packet4f& a, const Packet4f& b)
{
  Packet4f t, y_0, y_1, res;

  // Altivec does not offer a divide instruction, we have to do a reciprocal approximation
  y_0 = __vrefp(b);

  // Do one Newton-Raphson iteration to get the needed accuracy
  t   = __vnmsubfp(y_0, b, p4f_ONE);
  y_1 = __vmaddfp(y_0, t, y_0);

  res = __vmulfp(a, y_1);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet4i pdiv<Packet4i>(const Packet4i& /*a*/, const Packet4i& /*b*/)
{ eigen_assert(false && "packet integer division are not supported by AltiVec");
  return pset1<Packet4i>(0);
}

// for some weird raisons, it has to be overloaded for packet of integers
template<> EIGEN_STRONG_INLINE Packet4f pmadd(const Packet4f& a, const Packet4f& b, const Packet4f& c) { return __vmaddfp(a, b, c); }
template<> EIGEN_STRONG_INLINE Packet4i pmadd(const Packet4i& a, const Packet4i& b, const Packet4i& c) { return padd(pmul(a,b), c); }

template<> EIGEN_STRONG_INLINE Packet4f pmin<Packet4f>(const Packet4f& a, const Packet4f& b) { return __vminfp(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pmin<Packet4i>(const Packet4i& a, const Packet4i& b) { return __vminsw(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pmax<Packet4f>(const Packet4f& a, const Packet4f& b) { return __vmaxfp(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pmax<Packet4i>(const Packet4i& a, const Packet4i& b) { return __vmaxsw(a, b); }

// Logical Operations are not supported for float, so we have to reinterpret casts using NEON intrinsics
template<> EIGEN_STRONG_INLINE Packet4f pand<Packet4f>(const Packet4f& a, const Packet4f& b) { return __vand(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pand<Packet4i>(const Packet4i& a, const Packet4i& b) { return __vand(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f por<Packet4f>(const Packet4f& a, const Packet4f& b) { return __vor(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i por<Packet4i>(const Packet4i& a, const Packet4i& b) { return __vor(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pxor<Packet4f>(const Packet4f& a, const Packet4f& b) { return __vxor(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pxor<Packet4i>(const Packet4i& a, const Packet4i& b) { return __vxor(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pandnot<Packet4f>(const Packet4f& a, const Packet4f& b) { return __vandc(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pandnot<Packet4i>(const Packet4i& a, const Packet4i& b) { return __vandc(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pload<Packet4f>(const float* from) { EIGEN_DEBUG_ALIGNED_LOAD return __lvx (from, 0); }
template<> EIGEN_STRONG_INLINE Packet4i pload<Packet4i>(const int*     from) { EIGEN_DEBUG_ALIGNED_LOAD return __lvx (from, 0); }

template<> EIGEN_STRONG_INLINE Packet4f ploadu<Packet4f>(const float* from) { EIGEN_DEBUG_UNALIGNED_LOAD return __vor(__lvlx(from, 0), __lvrx(from, 16)); }
template<> EIGEN_STRONG_INLINE Packet4i ploadu<Packet4i>(const int* from) { EIGEN_DEBUG_UNALIGNED_LOAD return __vor(__lvlx(from, 0), __lvrx(from, 16)); }

template<> EIGEN_STRONG_INLINE Packet4f ploaddup<Packet4f>(const float*   from)
{
  Packet4f p;
  if((ptrdiff_t(from) % 16) == 0)   p = pload<Packet4f>(from);
  else                              p = ploadu<Packet4f>(from);
  return __vpermwi(p, pic_DUPLICATE);
}
template<> EIGEN_STRONG_INLINE Packet4i ploaddup<Packet4i>(const int*     from)
{
  Packet4i p;
  if((ptrdiff_t(from) % 16) == 0)   p = pload<Packet4i>(from);
  else                              p = ploadu<Packet4i>(from);
  return __vpermwi(p, pic_DUPLICATE);
}

template<> EIGEN_STRONG_INLINE void pstore<float>(float*   to, const Packet4f& from) { EIGEN_DEBUG_ALIGNED_STORE __stvx(from, to, 0); }
template<> EIGEN_STRONG_INLINE void pstore<int>(int*       to, const Packet4i& from) { EIGEN_DEBUG_ALIGNED_STORE __stvx(from, to, 0); }

template<> EIGEN_STRONG_INLINE void pstoreu<float>(float*  to, const Packet4f& from)
{
  EIGEN_DEBUG_UNALIGNED_STORE
  __stvlx(from, to, 0);
  __stvrx(from, to, 16);
}
template<> EIGEN_STRONG_INLINE void pstoreu<int>(int*      to, const Packet4i& from)
{
  EIGEN_DEBUG_UNALIGNED_STORE
  __stvlx(from, to, 0);
  __stvrx(from, to, 16);
}

template<> EIGEN_STRONG_INLINE void prefetch<float>(const float* addr) { __dcbt(0, addr); }
template<> EIGEN_STRONG_INLINE void prefetch<int>(const int*     addr) { __dcbt(0, addr); }

template<> EIGEN_STRONG_INLINE float  pfirst<Packet4f>(const Packet4f& a) { float EIGEN_ALIGN16 x[4]; __stvewx(a, x, 0); return x[0]; }
template<> EIGEN_STRONG_INLINE int    pfirst<Packet4i>(const Packet4i& a) { int   EIGEN_ALIGN16 x[4]; __stvewx(a, x, 0); return x[0]; }

template<> EIGEN_STRONG_INLINE Packet4f preverse(const Packet4f& a) { return (Packet4f)__vpermwi(a, pic_REVERSE); }
template<> EIGEN_STRONG_INLINE Packet4i preverse(const Packet4i& a) { return (Packet4i)__vpermwi(a, pic_REVERSE); }

template<> EIGEN_STRONG_INLINE Packet4f pabs(const Packet4f& a) { return pmax(a, pnegate(a)); }
template<> EIGEN_STRONG_INLINE Packet4i pabs(const Packet4i& a) { return pmax(a, pnegate(a)); }

template<> EIGEN_STRONG_INLINE float predux<Packet4f>(const Packet4f& a)
{
  Packet4f sum = __vmsum4fp(a, p4f_ONE);
  return pfirst(sum);
}

template<> EIGEN_STRONG_INLINE Packet4f preduxp<Packet4f>(const Packet4f* vecs)
{
#if 0 // reference translation
  Packet4f v[4], sum[4];

  // It's easier and faster to transpose then add as columns
  // Check: http://www.freevec.org/function/matrix_4x4_transpose_floats for explanation
  // Do the transpose, first set of moves
  v[0] = __vmrghw(vecs[0], vecs[2]);
  v[1] = __vmrglw(vecs[0], vecs[2]);
  v[2] = __vmrghw(vecs[1], vecs[3]);
  v[3] = __vmrglw(vecs[1], vecs[3]);
  // Get the resulting vectors
  sum[0] = __vmrghw(v[0], v[2]);
  sum[1] = __vmrglw(v[0], v[2]);
  sum[2] = __vmrghw(v[1], v[3]);
  sum[3] = __vmrglw(v[1], v[3]);

  // Now do the summation:
  // Lines 0+1
  sum[0] = __vaddfp(sum[0], sum[1]);
  // Lines 2+3
  sum[1] = __vaddfp(sum[2], sum[3]);
  // Add the results
  sum[0] = __vaddfp(sum[0], sum[1]);
#else // using dot products
  Packet4f sum[4];
  
  sum[0] = __vmsum4fp(vecs[0], vecs[0]);
  sum[1] = __vmsum4fp(vecs[1], vecs[1]);
  sum[2] = __vmsum4fp(vecs[2], vecs[2]);
  sum[3] = __vmsum4fp(vecs[3], vecs[3]);

  sum[0] = __vsel(sum[0], sum[1], p4i_SEL_STEP1);
  sum[1] = __vsel(sum[2], sum[3], p4i_SEL_STEP1);
  sum[0] = __vsel(sum[0], sum[1], p4i_SEL_STEP2);
#endif

  return sum[0];
}

template<> EIGEN_STRONG_INLINE int predux<Packet4i>(const Packet4i& a)
{
  Packet4i sum = __vaddsws(a, __vpermwi(a, pic_REV2));
  sum = __vaddsws(sum, __vpermwi(sum, pic_REV));
  return pfirst(sum);
}

template<> EIGEN_STRONG_INLINE Packet4i preduxp<Packet4i>(const Packet4i* vecs)
{
  Packet4i v[4], sum[4];

  // It's easier and faster to transpose then add as columns
  // Check: http://www.freevec.org/function/matrix_4x4_transpose_floats for explanation
  // Do the transpose, first set of moves
  v[0] = __vmrghw(vecs[0], vecs[2]);
  v[1] = __vmrglw(vecs[0], vecs[2]);
  v[2] = __vmrghw(vecs[1], vecs[3]);
  v[3] = __vmrglw(vecs[1], vecs[3]);
  // Get the resulting vectors
  sum[0] = __vmrghw(v[0], v[2]);
  sum[1] = __vmrglw(v[0], v[2]);
  sum[2] = __vmrghw(v[1], v[3]);
  sum[3] = __vmrglw(v[1], v[3]);

  // Now do the summation:
  // Lines 0+1
  sum[0] = __vaddsws(sum[0], sum[1]);
  // Lines 2+3
  sum[1] = __vaddsws(sum[2], sum[3]);
  // Add the results
  sum[0] = __vaddsws(sum[0], sum[1]);

  return sum[0];
}

// Other reduction functions:
// mul
template<> EIGEN_STRONG_INLINE float predux_mul<Packet4f>(const Packet4f& a)
{
  Packet4f prod;
  prod = __vmulfp(a, __vsldoi(a, a, 8));
  return pfirst((Packet4f)__vmulfp(prod, __vsldoi(prod, prod, 4)));
}

template<> EIGEN_STRONG_INLINE int predux_mul<Packet4i>(const Packet4i& a)
{
  EIGEN_ALIGN16 int aux[4];
  pstore(aux, a);
  return aux[0] * aux[1] * aux[2] * aux[3];
}

// min
template<> EIGEN_STRONG_INLINE float predux_min<Packet4f>(const Packet4f& a)
{
  Packet4f b, res;
  b = __vminfp(a, __vsldoi(a, a, 8));
  res = __vminfp(b, __vsldoi(b, b, 4));
  return pfirst(res);
}

template<> EIGEN_STRONG_INLINE int predux_min<Packet4i>(const Packet4i& a)
{
  Packet4i b, res;
  b = __vminsw(a, __vsldoi(a, a, 8));
  res = __vminsw(b, __vsldoi(b, b, 4));
  return pfirst(res);
}

// max
template<> EIGEN_STRONG_INLINE float predux_max<Packet4f>(const Packet4f& a)
{
  Packet4f b, res;
  b = __vmaxfp(a, __vsldoi(a, a, 8));
  res = __vmaxfp(b, __vsldoi(b, b, 4));
  return pfirst(res);
}

template<> EIGEN_STRONG_INLINE int predux_max<Packet4i>(const Packet4i& a)
{
  Packet4i b, res;
  b = __vmaxsw(a, __vsldoi(a, a, 8));
  res = __vmaxsw(b, __vsldoi(b, b, 4));
  return pfirst(res);
}

template<int Offset>
struct palign_impl<Offset,Packet4f>
{
  static EIGEN_STRONG_INLINE void run(Packet4f& first, const Packet4f& second)
  {
    if (Offset!=0)
      first = __vsldoi(first, second, Offset*4);
  }
};

template<int Offset>
struct palign_impl<Offset,Packet4i>
{
  static EIGEN_STRONG_INLINE void run(Packet4i& first, const Packet4i& second)
  {
    if (Offset!=0)
      first = __vsldoi(first, second, Offset*4);
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PACKET_MATH_ALTIVEC_H
