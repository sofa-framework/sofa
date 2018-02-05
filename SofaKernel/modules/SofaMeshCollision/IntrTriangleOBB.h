/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_INTRTRIANGLEOBB_H
#define SOFA_COMPONENT_COLLISION_INTRTRIANGLEOBB_H
#include "config.h"

#include <sofa/core/collision/Intersection.h>
#include <SofaBaseCollision/OBBModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/IntrMeshUtility.h>
#include <SofaBaseCollision/Intersector.h>

namespace sofa{
namespace component{
namespace collision{

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
    typedef defaulttype::Vec<3,Real> Vec3;

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

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_INTRTRIANGLEOBB_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MESH_COLLISION_API TIntrTriangleOBB<defaulttype::Vec3dTypes,defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MESH_COLLISION_API TIntrTriangleOBB<defaulttype::Vec3fTypes,defaulttype::Rigid3fTypes>;
#endif
#endif

}
}
}

#endif // SOFA_COMPONENT_COLLISION_INTRTRIANGLEOBB_H
