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
* File: ./src/librender/bbox.cpp                                  *
*                                                                 *
* Contacts:                                                       *
*                                                                 *
******************************************************************/
#include <flowvr/render/bbox.h>

#include <limits>
#include <algorithm>
#include <sstream>

using std::min;
using std::max;
using namespace ftl;

namespace flowvr
{

namespace render
{

static const float Fmax = std::numeric_limits<float>::max();
static const float Fmin = std::numeric_limits<float>::min();

BBox::BBox()
  : a(Fmax,Fmax,Fmax), b(-Fmax,-Fmax,-Fmax)
{
}

BBox::BBox(const Vec3f& _a, const Vec3f& _b)
  : a(_a), b(_b)
{
}

void BBox::clear()
{
  a.fill(Fmax);
  b.fill(-Fmax);
}

bool BBox::isEmpty() const
{
  return (a.x() > b.x());
}

BBox& BBox::operator+=(const BBox &p) 
{
  if (p.a.x()<a.x()) a.x()=p.a.x();
  if (p.a.y()<a.y()) a.y()=p.a.y();
  if (p.a.z()<a.z()) a.z()=p.a.z();
  if (p.b.x()>b.x()) b.x()=p.b.x();
  if (p.b.y()>b.y()) b.y()=p.b.y();
  if (p.b.z()>b.z()) b.z()=p.b.z();
  return *this;
}

BBox& BBox::operator+=(const Vec3f &p)
{
  if (p.x()<a.x()) a.x()=p.x();
  if (p.x()>b.x()) b.x()=p.x();
  if (p.y()<a.y()) a.y()=p.y();
  if (p.y()>b.y()) b.y()=p.y();
  if (p.z()<a.z()) a.z()=p.z();
  if (p.z()>b.z()) b.z()=p.z();
  return *this;
}

BBox BBox::operator+(const BBox &p)
{
  return BBox(Vec3f(min(a.x(),p.a.x())
		    ,min(a.y(),p.a.y())
		    ,min(a.z(),p.a.z()))
	      ,Vec3f(max(b.x(),p.b.x())
		     ,max(b.y(),p.b.y())
		     ,max(b.z(),p.b.z()))
  );
}

void BBox::apply(const BBox &p,const Mat4x4f &m)
{
  if (p.isEmpty())
  {
    clear();
  }
  else
  {
    for (int l=0;l<3;l++)
    {
      a[l] = b[l] = m[l][3];
      for (int c=0;c<3;c++)
      {
	if (m[l][c]>0.0f) { a[l]+=m[l][c]*p.a[c]; b[l]+=m[l][c]*p.b[c]; }
	else              { a[l]+=m[l][c]*p.b[c]; b[l]+=m[l][c]*p.a[c]; }
      }
    }
  }
}

BBox BBox::operator*(const Mat4x4f &m)
{
  BBox r;
  r.apply(*this,m);
  return r;
}

/// Test if bbox intersect with half-space defined as eq.P >= 0
bool BBox::inHalfspace(const ftl::Vec4f& eq) const
{
  Vec3f p;
  if (eq[0]>=0) p[0]=b[0]; else p[0]=a[0];
  if (eq[1]>=0) p[1]=b[1]; else p[1]=a[1];
  if (eq[2]>=0) p[2]=b[2]; else p[2]=a[2];
  return (eq[0]*p[0]+eq[1]*p[1]+eq[2]*p[2]+eq[3])>=0;
}

/// Test if bbox intersect with frustum defined by the given matrix
/// (interpreted as an OpenGL projection matrix)
bool BBox::inFrustum(const ftl::Mat4x4f& frustum, bool exact) const
{
  if (isEmpty()) return false;
  if (exact)
  { // axis-aligned tests
    Mat3x3f m0;
    m0(0,0)=frustum(0,0);  m0(0,1)=frustum(0,1);  m0(0,2)=frustum(0,2);
    m0(1,0)=frustum(1,0);  m0(1,1)=frustum(1,1);  m0(1,2)=frustum(1,2);
    m0(2,0)=frustum(2,0);  m0(2,1)=frustum(2,1);  m0(2,2)=frustum(2,2);
    Mat3x3f minv; minv.invert(m0);
    // m0.p0 + frustum.col(3) = 0;
    // p0 = minv.(-frustum.col(3));
    Vec3f p0;
    p0 = minv*Vec3f(-frustum(0,3),-frustum(1,3),-frustum(2,3));

    // p1 = inv.( 1, 1,1)
    // p2 = inv.(-1, 1,1)
    // p3 = inv.( 1,-1,1)
    // p4 = inv.(-1,-1,1)
    // if p*[C] >= 0 then bb.b[C] >= p0[C]
    // if p*[C] <= 0 then bb.a[C] <= p0[C]
    const Mat4x3f pmat(Vec3f(1,1,1),Vec3f(-1,1,1),Vec3f(1,-1,1),Vec3f(-1,-1,1));

    for (int c=0;c<3;c++)
    {
      Vec4f p = pmat*minv[c];
      if (p[0]>=0 && p[1]>=0 && p[2]>=0 && p[3]>=0)
      {
	if (b[c]<p0[c]) return false;
      }
      else if (p[0]<=0 && p[1]<=0 && p[2]<=0 && p[3]<=0)
      {
	if (a[c]>p0[c]) return false;
      }
    }
  }
  // halfplanes from frustum
  return inHalfspace(frustum[3]) // W>=0
    &&   inHalfspace(frustum[3]-frustum[0]) // X<=W  : W-X>=0
    &&   inHalfspace(frustum[3]-frustum[1]) // Y<=W  : W-Y>=0
    &&   inHalfspace(frustum[3]+frustum[0]) // X>=-W : W+X>=0
    &&   inHalfspace(frustum[3]+frustum[1]) // Y>=-W : W+Y>=0
    ;
}

std::string BBox::toString() const
{
  std::ostringstream os;
  os << *this;
  return os.str();
}

void BBox::fromString(const std::string& text)
{
  std::istringstream is(text);
  is >> *this;
}

} // namespace render

} // namespace flowvr


// iostream

std::ostream& operator<<(std::ostream& o, const flowvr::render::BBox& bb)
{
  o << bb.a << '-' << bb.b;
  return o;
}

std::istream& operator>>(std::istream& in, flowvr::render::BBox& bb)
{
  in >> bb.a;
  if (in.peek()=='-') in.get();
  in >> bb.b;
  return in;
}

