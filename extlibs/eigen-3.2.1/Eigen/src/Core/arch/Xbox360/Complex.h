// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_ALTIVEC_H
#define EIGEN_COMPLEX_ALTIVEC_H

namespace Eigen {

namespace internal {

static Packet4ui  p4ui_CONJ_XOR = (Packet4ui)__vmrghw(p4i_ZERO, p4f_ZERO_);//{ 0x00000000, 0x80000000, 0x00000000, 0x80000000 };

const unsigned int pic_COMPLEX_RE = _EIGEN_PERMUTATION_MASK(0,0,2,2);
const unsigned int pic_COMPLEX_IM = _EIGEN_PERMUTATION_MASK(1,1,3,3);
const unsigned int pic_PSET_HI = _EIGEN_PERMUTATION_MASK(0,1,0,1);
const unsigned int pic_PSET_LO = _EIGEN_PERMUTATION_MASK(2,3,2,3);

//---------- float ----------
__declspec(passinreg) struct Packet2cf : __vector4
{
	EIGEN_STRONG_INLINE Packet2cf() {}
	EIGEN_STRONG_INLINE Packet2cf(__vector4 v_): __vector4(v_) {}
};

template<> struct packet_traits<std::complex<float> >  : default_packet_traits
{
  typedef Packet2cf type;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 2,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0
  };
};

template<> struct unpacket_traits<Packet2cf> { typedef std::complex<float> type; enum {size=2}; };

template<> EIGEN_STRONG_INLINE Packet2cf pset1<Packet2cf>(const std::complex<float>&  from)
{
  Packet2cf res;
  /* On AltiVec we cannot load 64-bit registers, so wa have to take care of alignment */
  if((ptrdiff_t(&from) % 16) == 0)
    res = pload<Packet4f>((const float *)&from);
  else
    res = ploadu<Packet4f>((const float *)&from);
  res = __vpermwi(res, pic_PSET_HI);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet2cf padd<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return __vaddfp(a,b); }
template<> EIGEN_STRONG_INLINE Packet2cf psub<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return __vsubfp(a,b); }
template<> EIGEN_STRONG_INLINE Packet2cf pnegate(const Packet2cf& a) { return __vsubfp(p4f_ZERO, a); }
template<> EIGEN_STRONG_INLINE Packet2cf pconj(const Packet2cf& a) { return __vxor(a, p4ui_CONJ_XOR); }

template<> EIGEN_STRONG_INLINE Packet2cf pmul<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  Packet v1, v2;

  // Permute and multiply the real parts of a and b
  v1 = __vpermwi(a, pic_COMPLEX_RE);
  // Get the imaginary parts of a
  v2 = __vpermwi(a, pic_COMPLEX_IM);
  // multiply a_re * b 
  v1 = __vmulfp(v1, b);
  // multiply a_im * b and get the conjugate result
  v2 = __vmulfp(v2, b);
  v2 = __vxor(v2, p4ui_CONJ_XOR);
  // permute back to a proper order
  v2 = __vpermwi(v2, pic_REV);
  
  return __vaddfp(v1, v2);
}

template<> EIGEN_STRONG_INLINE Packet2cf pand   <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return __vand(a,b); }
template<> EIGEN_STRONG_INLINE Packet2cf por    <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return __vor(a,b); }
template<> EIGEN_STRONG_INLINE Packet2cf pxor   <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return __vxor(a,b); }
template<> EIGEN_STRONG_INLINE Packet2cf pandnot<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return __vandc(a, b); }

template<> EIGEN_STRONG_INLINE Packet2cf pload <Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_ALIGNED_LOAD return __lvx (from, 0); }
template<> EIGEN_STRONG_INLINE Packet2cf ploadu<Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_UNALIGNED_LOAD return __vor(__lvlx(from, 0), __lvrx(from, 16)); }

template<> EIGEN_STRONG_INLINE Packet2cf ploaddup<Packet2cf>(const std::complex<float>* from)
{
  return pset1<Packet2cf>(*from);
}

template<> EIGEN_STRONG_INLINE void pstore <std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_ALIGNED_STORE __stvx(from, to, 0); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_UNALIGNED_STORE __stvlx(from, to, 0); __stvrx(from, to, 16); }

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<float> >(const std::complex<float> *   addr) {  __dcbt(0, addr); }

template<> EIGEN_STRONG_INLINE std::complex<float>  pfirst<Packet2cf>(const Packet2cf& a)
{
  std::complex<float> EIGEN_ALIGN16 res[2];
  pstore(res, a);

  return res[0];
}

template<> EIGEN_STRONG_INLINE Packet2cf preverse(const Packet2cf& a)
{
  return __vpermwi(a, pic_REV2);
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux<Packet2cf>(const Packet2cf& a)
{
  Packet2cf b = __vsldoi(a, a, 8);
  b = padd(a, b);
  return pfirst(b);
}

template<> EIGEN_STRONG_INLINE Packet2cf preduxp<Packet2cf>(const Packet2cf* vecs)
{
  Packet b1, b2;
  
  b1 = __vsldoi(vecs[0], vecs[1], 8);
  b2 = __vsldoi(vecs[1], vecs[0], 8);
  b2 = __vsldoi(b2, b2, 8);
  b2 = __vaddfp(b1, b2);

  return b2;
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux_mul<Packet2cf>(const Packet2cf& a)
{
  Packet2cf b, prod;
  b = __vsldoi(a, a, 8);
  prod = pmul(a, b);

  return pfirst(prod);
}

template<int Offset>
struct palign_impl<Offset,Packet2cf>
{
  static EIGEN_STRONG_INLINE void run(Packet2cf& first, const Packet2cf& second)
  {
    if (Offset==1)
    {
      first = __vsldoi(first, second, 8);
    }
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, false,true>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    return internal::pmul(a, pconj(b));
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, true,false>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    return internal::pmul(pconj(a), b);
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, true,true>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    return pconj(internal::pmul(a, b));
  }
};

template<> EIGEN_STRONG_INLINE Packet2cf pdiv<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  // TODO optimize it for AltiVec
  Packet2cf res = conj_helper<Packet2cf,Packet2cf,false,true>().pmul(a,b);
  Packet s = __vmulfp(b, b);
  return Packet2cf(pdiv(res, (Packet2cf) __vaddfp(s,__vpermwi(s, pic_REV))));
}

template<> EIGEN_STRONG_INLINE Packet2cf pcplxflip<Packet2cf>(const Packet2cf& x)
{
  return Packet2cf(__vpermwi(x, pic_REV));
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COMPLEX_ALTIVEC_H
