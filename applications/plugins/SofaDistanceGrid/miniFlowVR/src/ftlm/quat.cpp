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
* File: src/ftlm/quat.cpp                                         *
*                                                                 *
* Contacts: 20/09/2005 Clement Menier <clement.menier.fr>         *
*                                                                 *
******************************************************************/
#include "ftl/quat.h"
#include <math.h>

#define PI 3.14159265358979323846264338327

namespace ftl
{

Quat & Quat::operator+=(const Quat &q)
{
  w+=q.w;
  x+=q.x;
  y+=q.y;
  z+=q.z;
  return *this;
}
Quat & Quat::operator-=(const Quat &q)
{
  w-=q.w;
  x-=q.x;
  y-=q.y;
  z-=q.z;
  return *this;
}
Quat & Quat::operator*=(const Quat &q)
{
  return *this=(*this)*q;
}

Quat & Quat::operator*=(float f)
{
  w*=f;
  x*=f;
  y*=f;
  z*=f;
  return *this;
}
Quat & Quat::operator/=(float f)
{
  w/=f;
  x/=f;
  y/=f;
  z/=f;
  return *this;
}

Quat Quat::operator+(const Quat &q) const
{
  return Quat(
	      w+q.w
	      ,x+q.x
	      ,y+q.y
	      ,z+q.z
	      );
}
Quat Quat::operator-(const Quat &q) const
{
  return Quat(
	      w-q.w
	      ,x-q.x
	      ,y-q.y
	      ,z-q.z
	      );
}
Quat Quat::operator*(const Quat &q) const
{
  return Quat(
	      w*q.w-x*q.x-y*q.y-z*q.z
	      ,w*q.x+x*q.w+y*q.z-z*q.y
	      ,w*q.y+y*q.w+z*q.x-x*q.z
	      ,w*q.z+z*q.w+x*q.y-y*q.x
	      );
}

Quat Quat::operator*(float f) const
{
  return Quat(
	      w*f
	      ,x*f
	      ,y*f
	      ,z*f
	      );
}
Quat Quat::operator/(float f) const
{
  return Quat(
	      w/f
	      ,x/f
	      ,y/f
	      ,z/f
	      );
}

Quat operator*(float f,const Quat &q)
{
  return Quat(
	      f*q.w
	      ,f*q.x
	      ,f*q.y
	      ,f*q.z
	      );
}

Quat Quat::operator-() const
{
  return Quat(
	      -w
	      ,-x
	      ,-y
	      ,-z
	      );
}

Quat Quat::operator~() const
{
  return Quat(
	      w
	      ,-x
	      ,-y
	      ,-z
	      );
}

void Quat::fromAngAxis(float ang,Vec3f axis)
{
// quaternions can represent a rotation.  The rotation is an angle t, around a 
// unit vector u.   q=(s,v);  s= cos(t/2);   v= u*sin(t/2).
  axis.normalize();

  w=(float)cos(ang/2);
  float f=(float)sin(ang/2);
  x=axis.x()*f;
  y=axis.y()*f;
  z=axis.z()*f;
}

void Quat::toAngAxis(float *ang,Vec3f *axis) const
{
  *ang = (float)acos (w) * 2;
  axis->x() = x;
  axis->y() = y;
  axis->z() = z;
  axis->normalize();
}

void Quat::fromDegreeAngAxis(float ang,Vec3f axis)
{
  return fromAngAxis((float)(ang*PI/180.0f), axis);
}
void Quat::toDegreeAngAxis(float *ang,Vec3f *axis) const
{
  toAngAxis(ang, axis);
  *ang *= (float)(180.0f/PI);
}

void Quat::fromMatrix(const Mat3x3f &m)
{
  float   tr, s;

  tr = m.x().x() + m.y().y() + m.z().z();

  // check the diagonal
  if (tr > 0)
  {
    s = (float)sqrt (tr + 1);
    w = s * 0.5f; // w OK
    s = 0.5f / s;
    x = (m.y().z() - m.z().y()) * s; // x OK
    y = (m.z().x() - m.x().z()) * s; // y OK
    z = (m.x().y() - m.y().x()) * s; // z OK
  }
  else
  {
    if (m.y().y() > m.x().x() && m.z().z() <= m.y().y())
    {
      s = (float)sqrt ((m.y().y() - (m.z().z() + m.x().x())) + 1.0f);

      y = s * 0.5f; // y OK

      if (s != 0.0f)
        s = 0.5f / s;

      z = (m.z().y() + m.y().z()) * s; // z OK
      x = (m.y().x() + m.x().y()) * s; // x OK
      w = (m.z().x() - m.x().z()) * s; // w OK
    }
    else if ((m.y().y() <= m.x().x()  &&  m.z().z() > m.x().x())  ||  (m.z().z() > m.y().y()))
    {
      s = (float)sqrt ((m.z().z() - (m.x().x() + m.y().y())) + 1.0f);

      z = s * 0.5f; // z OK

      if (s != 0.0f)
        s = 0.5f / s;

      x = (m.x().z() + m.z().x()) * s; // x OK
      y = (m.z().y() + m.y().z()) * s; // y OK
      w = (m.x().y() - m.y().x()) * s; // w OK
    }
    else
    {
      s = (float)sqrt ((m.x().x() - (m.y().y() + m.z().z())) + 1.0f);

      x = s * 0.5f; // x OK

      if (s != 0.0f)
          s = 0.5f / s;

      y = (m.y().x() + m.x().y()) * s; // y OK
      z = (m.x().z() + m.z().x()) * s; // z OK
      w = (m.y().z() - m.z().y()) * s; // w OK
    }
  }
}

void Quat::toMatrix(Mat3x3f *m) const
{
  float wx, wy, wz, xx, yy, yz, xy, xz, zz;

  xx = 2 * x * x;   xy = 2 * x * y;   xz = 2 * x * z;
  yy = 2 * y * y;   yz = 2 * y * z;   zz = 2 * z * z;
  wx = 2 * w * x;   wy = 2 * w * y;   wz = 2 * w * z;

  m->x().x() = 1 - yy - zz;  m->y().x() = xy - wz;      m->z().x() = xz + wy;
  m->x().y() = xy + wz;      m->y().y() = 1 - xx - zz;  m->z().y() = yz - wx;
  m->x().z() = xz - wy;      m->y().z() = yz + wx;      m->z().z() = 1 - xx - yy;
}

float Quat::length() const
{
  return (float)sqrt(w*w+x*x+y*y+z*z);
}

void Quat::clear()
{
  w=1;
  x=0;
  y=0;
  z=0;
}

void Quat::normalize()
{
  float f=length();
  if (f>0)
  {
    f=1/f;
    w*=f; x*=f; y*=f; z*=f;
  }
}

bool Quat::isIdentity() const
{
  return w>=1.0f;
}

// Assignement from typed data

namespace Type
{

template <>
bool assign(Quat& dest, int type, const void* data)
{
  Vec<4,float> tmp(1,0,0,0);
  if (!assign(tmp,type,data)) return false;
  dest.w = tmp[0];
  dest.x = tmp[1];
  dest.y = tmp[2];
  dest.z = tmp[3];
  dest.normalize();
  return true;
}

} // namespace Type

} // namespace ftl

std::ostream& operator<<(std::ostream& o, const ftl::Quat& q)
{
  o << ftl::Vec<4,float>(q.w,q.x,q.y,q.z);
  return o;
}

std::istream& operator>>(std::istream& in, ftl::Quat& q)
{
  ftl::Vec4f v;
  in >> v;
  q.w = v[0];
  q.x = v[1];
  q.y = v[2];
  q.z = v[3];
  return in;
}
