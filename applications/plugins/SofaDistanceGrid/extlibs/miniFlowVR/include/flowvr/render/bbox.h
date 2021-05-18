/******* COPYRIGHT ************************************************
*                                                                 *
*                         FlowVR Render                           *
*                   Parallel Rendering Library                    *
*                                                                 *
*-----------------------------------------------------------------*
* COPYRIGHT (C) 2005 by                                           *
* Laboratoire Informatique et Distribution (UMR5132) and          *
* INRIA Project MOVI. ALL RIGHTS RESERVED.                        *
*                                                                 *
* This source is covered by the GNU LGPL, please refer to the     *
* COPYING-LIB file for further information.                       *
*                                                                 *
*-----------------------------------------------------------------*
*                                                                 *
*  Original Contributors:                                         *
*    Jeremie Allard,                                              *
*    Clement Menier.                                              *
*                                                                 * 
*******************************************************************
*                                                                 *
* File: ./include/flowvr/render/bbox.h                            *
*                                                                 *
* Contacts:                                                       *
*                                                                 *
******************************************************************/
#ifndef FLOWVR_RENDER_BBOX_H
#define FLOWVR_RENDER_BBOX_H

#include <ftl/vec.h>
#include <ftl/mat.h>

#include <iostream>

namespace flowvr
{

namespace render
{

class BBox
{
 public:
  ftl::Vec3f a,b;

  BBox();
  BBox(const ftl::Vec3f& _a,const ftl::Vec3f& _b);

  bool isEmpty() const;

  void clear();

  BBox& operator+=(const BBox& p);

  BBox& operator+=(const ftl::Vec3f& p);

  BBox operator+(const BBox& p);

  void apply(const BBox& p,const ftl::Mat4x4f& m);

  BBox operator*(const ftl::Mat4x4f& m);

  float size() const
  {
    if (isEmpty()) return 0;
    ftl::Vec3f v=b-a;
    return v.norm();
  }

  bool in(const ftl::Vec3f& p) const
  {
    return (p.x()>=a.x() && p.x()<=b.x() && p.y()>=a.y() && p.y()<=b.y() && p.z()>=a.z() && p.z()<=b.z());
  }

  bool in(const BBox& bb) const
  {
    return (bb.b.x()>=a.x() && bb.a.x()<=b.x() && bb.b.y()>=a.y() && bb.a.y()<=b.y() && bb.b.z()>=a.z() && bb.a.z()<=b.z());
  }

  /// Test if bbox intersect with half-space defined as eq.P >= 0
  bool inHalfspace(const ftl::Vec4f& eq) const;

  /// Test if bbox intersect with frustum defined by the given matrix
  /// (interpreted as an OpenGL projection matrix)
  /// if exact is false then the result is overestimated
  /// if exact is true then a 3x3 matrix inversion is required
  bool inFrustum(const ftl::Mat4x4f& frustum, bool exact=false) const;

  bool operator==(const BBox& p) const
  {
    return a==p.a && b==p.b;
  }

  bool operator!=(const BBox& p) const
  {
    return a!=p.a || b!=p.b;
  }

  std::string toString() const;
  void fromString(const std::string& text);

};

} // namespace render

} // namespace flowvr

// iostream
std::ostream& operator<<(std::ostream& o, const flowvr::render::BBox& b);
std::istream& operator>>(std::istream& in, flowvr::render::BBox& b);

#endif
