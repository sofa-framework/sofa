/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <CollisionOBBCapsule/config.h>

#include <sofa/core/collision/Intersection.h>
#include <CollisionOBBCapsule/geometry/OBBModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <CollisionOBBCapsule/detection/intersection/IntrMeshUtility.h>
#include <CollisionOBBCapsule/detection/intersection/Intersector.h>

namespace collisionobbcapsule::detection::intersection
{

/**
  *TDataTypes is the sphere type and TDataTypes2 the OBB type.
  */
template <class TDataTypes,class TDataTypes2>
class TIntrTriangleOBB : public Intersector<typename TDataTypes::Real>
{
public:
    typedef TTriangle<TDataTypes> IntrTri;
    typedef typename TDataTypes::Real Real;
    typedef typename IntrTri::Coord Coord;
    typedef TOBB<TDataTypes2> Box;
    typedef type::Vec<3,Real> Vec3;

    TIntrTriangleOBB (const IntrTri& tri, const Box & box);

    bool Find(Real tmax,int tri_flg);

    bool Find(Real tmax);
private:
    using Intersector<Real>::_is_colliding;
    using Intersector<Real>::_pt_on_first;
    using Intersector<Real>::_pt_on_second;
    using Intersector<Real>::mContactTime;
    using Intersector<Real>::_sep_axis;

    // The objects to intersect.
    const IntrTri* _tri;
    const Box * mBox;
};

typedef TIntrTriangleOBB<defaulttype::Vec3Types,defaulttype::Rigid3Types> IntrTriangleOBB;

#if  !defined(SOFA_COMPONENT_COLLISION_INTRTRIANGLEOBB_CPP)
extern template class COLLISIONOBBCAPSULE_API TIntrTriangleOBB<defaulttype::Vec3Types,defaulttype::Rigid3Types>;

#endif

} // namespace collisionobbcapsule::detection::intersection
