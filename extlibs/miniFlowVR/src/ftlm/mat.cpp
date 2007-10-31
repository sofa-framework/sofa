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
* File: src/ftlm/mat.cpp                                          *
*                                                                 *
* Contacts: 20/09/2005 Clement Menier <clement.menier.fr>         *
*                                                                 *
******************************************************************/
#include "ftl/mat.h"
#include "ftl/quat.h"
#include <math.h>

namespace ftl
{

Mat4x4f matrixTranslation(const Vec3f& pos)
{
  Mat4x4f m; m.identity();
  m[0][3] = pos[0];
  m[1][3] = pos[1];
  m[2][3] = pos[2];
  return m;
}

Mat4x4f matrixScale(const Vec3f& scale)
{
  Mat4x4f m; m.identity();
  m[0][0] = scale[0];
  m[1][1] = scale[1];
  m[2][2] = scale[2];
  return m;
}

Mat4x4f matrixScale(const float scale)
{
  Mat4x4f m; m.identity();
  m[0][0] = scale;
  m[1][1] = scale;
  m[2][2] = scale;
  return m;
}

Mat4x4f matrixRotation(const Quat& rot)
{
  Mat3x3f r; rot.toMatrix(&r);
  Mat4x4f m; m.identity();
  m = r;
  return m;
}

Mat4x4f matrixRotation(float ang, const Vec3f& axis)
{
  Quat q; q.fromDegreeAngAxis(ang, axis);
  return matrixRotation(q);
}


Mat4x4f matrixTransform(const Vec3f& pos, const Quat& rot, const Vec3f& scale)
{
  return matrixTranslation(pos) * matrixRotation(rot) * matrixScale(scale);
}

Mat4x4f matrixTransform(const Vec3f& pos, const Quat& rot, float scale)
{
  return matrixTranslation(pos) * matrixRotation(rot) * matrixScale(scale);
}

Mat4x4f matrixTransform(const Vec3f& pos, float ang, const Vec3f& axis, const Vec3f& scale)
{
  return matrixTranslation(pos) * matrixRotation(ang,axis) * matrixScale(scale);
}

Mat4x4f matrixTransform(const Vec3f& pos, float ang, const Vec3f& axis, float scale)
{
  return matrixTranslation(pos) * matrixRotation(ang,axis) * matrixScale(scale);
}

Mat4x4f matrixTransform(const Vec3f& pos, const Vec3f& scale)
{
  return matrixTranslation(pos) * matrixScale(scale);
}

Mat4x4f matrixTransform(const Vec3f& pos, float scale)
{
  return matrixTranslation(pos) * matrixScale(scale);
}

Mat4x4f matrixTransform(const Vec3f& pos)
{
  return matrixTranslation(pos);
}

} // namespace ftl
