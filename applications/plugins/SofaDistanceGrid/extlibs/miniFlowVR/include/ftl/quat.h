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
* File: include/ftl/quat.h                                        *
*                                                                 *
* Contacts: 20/09/2005 Clement Menier <clement.menier.fr>         *
*                                                                 *
******************************************************************/
#ifndef FTL_QUAT_H
#define FTL_QUAT_H

#include "vec.h"
#include "mat.h"

namespace ftl
{

// Basic angle-related definitions

static const double Pi = 3.1415926535897932384626433832795029;

extern inline double RAD2DEG(double a) { return a*180/Pi; }
extern inline double DEG2RAD(double a) { return a*Pi/180; }

/// Quaternion
class Quat
{
 public:

  float w,x,y,z;

  Quat() : w(1),x(0),y(0),z(0) {}
  Quat(float _w,float _x,float _y,float _z) : w(_w),x(_x),y(_y),z(_z) {}
  Quat(float _w,const Vec3f &_v) : w(_w), x(_v[0]),y(_v[1]),z(_v[2]) {}

  Quat & operator+=(const Quat &q);
  Quat & operator-=(const Quat &q);
  Quat & operator*=(const Quat &q);
  //Quat & operator/=(Quat &q);
  Quat & operator*=(float f);
  Quat & operator/=(float f);

  Quat operator+(const Quat &q) const;
  Quat operator-(const Quat &q) const;
  Quat operator*(const Quat &q) const;
  //Quat operator/(Quat &q);
  Quat operator*(float f) const;
  Quat operator/(float f) const;

  friend Quat operator*(float f,const Quat &q);

  Quat operator-() const;
  Quat operator~() const;

  void fromAngAxis(float ang,Vec3f axis);
  void toAngAxis(float *ang,Vec3f *axis) const;

  void fromDegreeAngAxis(float ang,Vec3f axis);
  void toDegreeAngAxis(float *ang,Vec3f *axis) const;

  void fromMatrix(const Mat3x3f &m);
  void toMatrix(Mat3x3f *m) const;

  float length() const;
  void clear();
  void normalize();
  bool isIdentity() const;

};

// Assignement from typed data

namespace Type
{

// template<> extern inline Type get(const Quat&) { return Vec4f; }

template <>
bool assign(Quat& dest, int type, const void* data);

} // namespace Type

} // namespace ftl

std::ostream& operator<<(std::ostream& o, const ftl::Quat& q);

std::istream& operator>>(std::istream& in, ftl::Quat& q);

#endif
