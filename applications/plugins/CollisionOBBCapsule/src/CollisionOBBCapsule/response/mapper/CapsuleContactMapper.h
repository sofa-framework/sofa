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

#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/RigidContactMapper.h>
#include <CollisionOBBCapsule/geometry/CapsuleModel.h>

using namespace sofa;
using namespace sofa::core::collision;
using sofa::component::collision::response::mapper::ContactMapper;
using sofa::component::collision::response::mapper::RigidContactMapper;
using collisionobbcapsule::geometry::CapsuleCollisionModel;

namespace sofa::component::collision::response::mapper
{


template <class DataTypes>
class COLLISIONOBBCAPSULE_API ContactMapper<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, DataTypes> : public BarycentricContactMapper<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, DataTypes> {
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    using Index = sofa::Index;

public:
    Index addPoint(const Coord& P, Index index, Real& r) {
        r = this->model->radius(index);

        SReal baryCoords[1];
        const Coord& p0 = this->model->point1(index);
        const Coord pA = this->model->point2(index) - p0;
        Coord pos = P - p0;
        baryCoords[0] = ((pos * pA) / pA.norm2());

        if (baryCoords[0] > 1)
            baryCoords[0] = 1;
        else if (baryCoords[0] < 0)
            baryCoords[0] = 0;

        return this->mapper->addPointInLine(index, baryCoords);
    }
};


template <class TVec3Types>
class COLLISIONOBBCAPSULE_API ContactMapper<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TVec3Types > : public RigidContactMapper<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TVec3Types > {
public:
    sofa::Index addPoint(const typename TVec3Types::Coord& P, sofa::Index index, typename TVec3Types::Real& r)
    {
        const typename TVec3Types::Coord& cP = P - this->model->center(index);
        const type::Quat<SReal>& ori = this->model->orientation(index);

        return RigidContactMapper<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TVec3Types >::addPoint(ori.inverseRotate(cP), index, r);
    }
};

#if !defined(SOFA_SOFAMISCCOLLISION_CAPSULECONTACTMAPPER_CPP)
extern template class COLLISIONOBBCAPSULE_API ContactMapper<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, sofa::defaulttype::Vec3Types>;
extern template class COLLISIONOBBCAPSULE_API ContactMapper<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, sofa::defaulttype::Vec3Types>;
#endif
} // namespace sofa::component::collision::response::mapper
