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

#include <sofa/component/collision/response/mapper/BaseContactMapper.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/RigidContactMapper.h>
#include <CollisionOBBCapsule/geometry/OBBModel.h>

using namespace sofa;
using namespace sofa::core::collision;
using sofa::component::collision::response::mapper::ContactMapper;
using sofa::component::collision::response::mapper::RigidContactMapper;
using collisionobbcapsule::geometry::OBBCollisionModel;

namespace sofa::component::collision::response::mapper
{

template <class TVec3Types>
class COLLISIONOBBCAPSULE_API ContactMapper<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, TVec3Types > : public RigidContactMapper<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, TVec3Types > {
public:
    sofa::Index addPoint(const typename TVec3Types::Coord& P, sofa::Index index, typename TVec3Types::Real& r)
    {
        const typename TVec3Types::Coord& cP = P - this->model->center(index);
        const type::Quat<SReal>& ori = this->model->orientation(index);

        return RigidContactMapper<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, TVec3Types >::addPoint(ori.inverseRotate(cP), index, r);
    }
};

#if !defined(SOFA_SOFAMISCCOLLISION_OBBCONTACTMAPPER_CPP)
extern template class response::mapper::ContactMapper<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, sofa::defaulttype::Vec3Types>;
#endif // 

} // namespace sofa::component::collision::response::mapper
