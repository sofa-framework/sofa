/******* COPYRIGHT ************************************************
*                                                                 *
*                             FlowVR                              *
*                       Template Library                          *
*                                                                 *
*-----------------------------------------------------------------*
* COPYRIGHT (C) 20054 by                                          *
* Laboratoire Informatique et Distribution (UMR5132) and          *
* INRIA Project MOVI. ALL RIGHTS RESERVED.                        *
*                                                                 *
* This source is covered by the GNU LGPL, please refer to the     *
* COPYING file for further information.                           *
*                                                                 *
*-----------------------------------------------------------------*
*                                                                 *
*  Original Contributors:                                         *
*    Jeremie Allard,                                              *
*    Clement Menier.                                              *
*                                                                 * 
*******************************************************************
*                                                                 *
* File: include/ftl/vec.h                                         *
*                                                                 *
* Contacts: 20/09/2005 Clement Menier <clement.menier.fr>         *
*                                                                 *
******************************************************************/
#ifndef FTL_VEC_H
#define FTL_VEC_H

#include <math.h>
#include "fixed_array.h"

#include "type.h"

#include <iostream>

namespace ftl
{

template <int N, typename real=float>
class Vec : public fixed_array<real,N>
{
public:

  static Type::Type getType() { return (Type::Type)Type::vector(Type::get(real()),N); }

  /// Default constructor: sets all values to 0.
  Vec()
  {
    this->assign(0);
  }

/*
  Vec(real r1)
  {
    static_assert(N == 1, "");
    this->elems[0]=r1;
  }
*/

  /// Specific constructor for 2-elements vectors.
  Vec(real r1, real r2)
  {
    static_assert(N == 2, "");
    this->elems[0]=r1;
    this->elems[1]=r2;
  }

  /// Specific constructor for 3-elements vectors.
  Vec(real r1, real r2, real r3)
  {
    static_assert(N == 3, "");
    this->elems[0]=r1;
    this->elems[1]=r2;
    this->elems[2]=r3;
  }

  /// Specific constructor for 4-elements vectors.
  Vec(real r1, real r2, real r3, real r4)
  {
    static_assert(N == 4, "");
    this->elems[0]=r1;
    this->elems[1]=r2;
    this->elems[2]=r3;
    this->elems[3]=r4;
  }

  /// Specific constructor for 5-elements vectors.
  Vec(real r1, real r2, real r3, real r4, real r5)
  {
    static_assert(N == 5, "");
    this->elems[0]=r1;
    this->elems[1]=r2;
    this->elems[2]=r3;
    this->elems[3]=r4;
    this->elems[4]=r5;
  }

  /// Specific constructor for 6-elements vectors (bounding-box).
  Vec(real r1, real r2, real r3, real r4, real r5, real r6)
  {
    static_assert(N == 6, "");
    this->elems[0]=r1;
    this->elems[1]=r2;
    this->elems[2]=r3;
    this->elems[3]=r4;
    this->elems[4]=r5;
    this->elems[5]=r6;
  }

  /// Constructor from an N-1 elements vector and an additional value (added at the end).
  Vec(const Vec<N-1,real>& v, real r1)
  {
    static_assert(N > 1, "");
    for(int i=0;i<N-1;i++)
      this->elems[i] = v[i];
    this->elems[N-1]=r1;
  }
  
  /// Constructor from an array of values.
  template<typename real2>
  explicit Vec(const real2* p)
  {
    std::copy(p, p+N, this->begin());
  }

  template<int M, typename real2>
  explicit Vec(const Vec<M, real2> &v)
  {
    std::copy(v.begin(), v.begin()+(N>M?M:N), this->begin());
  }

  /// Special access to first element.
  real& x() { static_assert(N >= 1, ""); return this->elems[0]; }
  /// Special access to second element.
  real& y() { static_assert(N >= 2, ""); return this->elems[1]; }
  /// Special access to third element.
  real& z() { static_assert(N >= 3, ""); return this->elems[2]; }
  /// Special access to fourth element.
  real& w() { static_assert(N >= 4, ""); return this->elems[3]; }

  /// Special const access to first element.
  const real& x() const { static_assert(N >= 1, ""); return this->elems[0]; }
  /// Special const access to second element.
  const real& y() const { static_assert(N >= 2, ""); return this->elems[1]; }
  /// Special const access to third element.
  const real& z() const { static_assert(N >= 3, ""); return this->elems[2]; }
  /// Special const access to fourth element.
  const real& w() const { static_assert(N >= 4, ""); return this->elems[3]; }
  
  /// Assignment operator from an array of values.
  void operator=(const real* p)
  {
    std::copy(p, p+N, this->begin());
  }

  /// Assignment from a vector with different dimensions.
template<int M, typename real2> void operator=(const Vec<M,real2>& v)
  {
    std::copy(v.begin(), v.begin()+(N>M?M:N), this->begin());
  }

  /// Sets every element to 0.
  void clear()
  {
    this->assign(0);
  }

  /// Sets every element to r.
  void fill(real r)
  {
    this->assign(r);
  }

  /// Access to i-th element.
  real& operator[](int i)
  {
    return this->elems[i];
  }

  /// Const access to i-th element.
  const real& operator[](int i) const
  {
    return this->elems[i];
  }

  /// Access to i-th element.
  real& operator()(int i)
  {
    return this->elems[i];
  }

  /// Const access to i-th element.
  const real& operator()(int i) const
  {
    return this->elems[i];
  }

  /// Cast into a const array of values.
  const real* ptr() const
  {
    return this->elems;
  }

  /// Cast into an array of values.
  real* ptr()
  {
    return this->elems;
  }

  // LINEAR ALGEBRA

  /// Multiplication by a scalar f.
  Vec<N,real> operator*(real f) const
  {
    Vec<N,real> r;
    for (int i=0;i<N;i++)
      r[i] = this->elems[i]*f;
    return r;
  }

  /// Scalar multiplication assignment operator.
  void operator*=(real f)
  {
    for (int i=0;i<N;i++)
      this->elems[i]*=f;
  }

  /// Division by a scalar f.
  Vec<N,real> operator/(real f) const
  {
    Vec<N,real> r;
    for (int i=0;i<N;i++)
      r[i] = this->elems[i]/f;
    return r;
  }

  /// On-place division by a scalar f.
  void operator/=(real f)
  {
    for (int i=0;i<N;i++)
      this->elems[i]/=f;
  }

  /// Dot product.
  real operator*(const Vec<N,real>& v) const
  {
    real r = this->elems[0]*v[0];
    for (int i=1;i<N;i++)
      r += this->elems[i]*v[i];
    return r;
  }

  /// Vector addition.
  Vec<N,real> operator+(const Vec<N,real>& v) const
  {
    Vec<N,real> r;
    for (int i=0;i<N;i++)
      r[i]=this->elems[i]+v[i];
    return r;
  }

  /// On-place vector addition.
  void operator+=(const Vec<N,real>& v)
  {
    for (int i=0;i<N;i++)
      this->elems[i]+=v[i];
  }

  /// Vector subtraction.
  Vec<N,real> operator-(const Vec<N,real>& v) const
  {
    Vec<N,real> r;
    for (int i=0;i<N;i++)
      r[i]=this->elems[i]-v[i];
    return r;
  }
  
  /// On-place vector subtraction.
  void operator-=(const Vec<N,real>& v)
  {
    for (int i=0;i<N;i++)
      this->elems[i]-=v[i];
  }

  /// Vector negation.
  Vec<N,real> operator-() const
  {
    Vec<N,real> r;
    for (int i=0;i<N;i++)
      r[i]=-this->elems[i];
    return r;
  }

  /// Squared norm.
  real norm2() const
  {
    real r = this->elems[0]*this->elems[0];
    for (int i=1;i<N;i++)
      r += this->elems[i]*this->elems[i];
    return r;
  }

  /// Euclidean norm.
  real norm() const
  {
    return sqrt(norm2());
  }

  /// Normalize the vector.
  void normalize()
  {
    real r = norm();
    if (r>1e-10)
    for (int i=0;i<N;i++)
      this->elems[i]/=r;
  }

};

/// Cross product for 3-elements vectors.
template<typename real>
inline Vec<3,real> cross(const Vec<3,real>& a, const Vec<3,real>& b)
{
  return Vec<3,real>(a.y()*b.z() - a.z()*b.y(),
		     a.z()*b.x() - a.x()*b.z(),
		     a.x()*b.y() - a.y()*b.x());
}

/// Dot product (alias for operator*)
template<int N,typename real>
inline real dot(const Vec<N,real>& a, const Vec<N,real>& b)
{
  return a*b;
}

// Definition of VecXY (X = 2,3,4) and Y = (b = byte, i = int, f = float, d = double).
typedef Vec<2,unsigned char> Vec2b;
typedef Vec<2,int> Vec2i;
typedef Vec<2,float> Vec2f;
typedef Vec<2,double> Vec2d;

typedef Vec<3,unsigned char> Vec3b;
typedef Vec<3,int> Vec3i;
typedef Vec<3,float> Vec3f;
typedef Vec<3,double> Vec3d;

typedef Vec<4,unsigned char> Vec4b;
typedef Vec<4,int> Vec4i;
typedef Vec<4,float> Vec4f;
typedef Vec<4,double> Vec4d;

// Typed data

namespace Type
{

//template<int N, class real>
//inline Type get(const Vec<N,real>& v)
//{
//  return (Type)vector(get(v[0]),N);
//}

/// Assign a vector from a string
template<int N, class real>
bool assignString(Vec<N,real>& dest, const std::string& data)
{
  const char* src = data.c_str();
  const char* end = src+data.length();
  bool res = true;
  int n = 0;
  while (n < N && src < end)
  {
    if (*src==' ' || *src==',') ++src;
    else
    {
      const char* s0 = src;
      if (n==0 && *s0 == '{') ++s0;
      do
	++src;
      while (src < end && *src != ' ' && *src != ',');
      int s = (int)(src-s0);
      if (src==end && src[-1]=='}') --s;
      if (s>0)
      {
	std::string str (s0, s0+s);
#ifdef DEBUG
	std::cerr << "dest["<<n<<"]=\""<<str<<"\""<<std::endl;
#endif
	res &= assign(dest[n], buildString(s), s0);
	++n;
      }
    }
  }
  return res;
}

template <int N, typename real>
class Assign< Vec<N,real> >
{
public:
static bool do_assign(Vec<N,real>& dest, int type, const void* data)
{
  if (isString(type))
  {
    return assignString(dest, std::string((const char*) data, size(type)));
  }
  else if (isVector(type))
  {
    int eSize = elemSize(type);
    int eType = toSingle(type);
    int n = nx(type);
    if (n>N) n=N;
    bool res = true;
    for (int i=0;i<n;i++)
      res &= assign(dest[i],eType,((const char*)data)+eSize*i);
    return res;
  }
  else
  {
    // Default implementation: read one value and fill the vector with it
    real r;
    if (!assign(r,type,data)) return false;
    dest.fill(r);
    return true;
  }
}
};

} // namespace Type

} // namespace ftl

// iostream

template <int N, typename real>
std::ostream& operator<<(std::ostream& o, const ftl::Vec<N,real>& v)
{
  o << '<' << v[0];
  for (int i=1; i<N; i++)
    o << ',' << v[i];
  o << '>';
  return o;
}

template <int N, typename real>
std::ostream& operator<<(std::ostream& o, ftl::Vec<N,real>& v)
{
  o << '<' << v[0];
  for (int i=1; i<N; i++)
    o << ',' << v[i];
  o << '>';
  return o;
}

template <int N, typename real>
std::istream& operator>>(std::istream& in, ftl::Vec<N,real>& v)
{
  int c;
  c = in.peek();
  while (c==' ' || c=='\n' || c=='<')
  {
    in.get();
    c = in.peek();
  }
  in >> v[0];
  for (int i=1; i<N; i++)
  {
    c = in.peek();
    while (c==' ' || c==',')
    {
      in.get();
      c = in.peek();
    }
    in >> v[i];
  }
  c = in.peek();
  while (c==' ' || c=='\n' || c=='>')
  {
    in.get();
    c = in.peek();
  }
  return in;
}

/// Scalar multiplication operator.
template <int N, typename real>
ftl::Vec<N,real> operator*(real r, const ftl::Vec<N,real>& v)
{
  return v*r;
}

#endif
