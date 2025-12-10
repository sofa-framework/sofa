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
#include <sofa/component/collision/response/mapper/config.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/collision/geometry/TetrahedronCollisionModel.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>

namespace sofa::component::collision::response::mapper
{

/// Mapper for TetrahedronCollisionModel
template<class DataTypes>
class ContactMapper<collision::geometry::TetrahedronCollisionModel, DataTypes> : public BarycentricContactMapper<collision::geometry::TetrahedronCollisionModel, DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    sofa::Index addPoint(const Coord& P, sofa::Index index, Real&)
    {
        collision::geometry::Tetrahedron t(this->model, index);
        auto b = t.getBary(P);
        return this->mapper->addPointInTetra(index, b.ptr());
    }
};

#if !defined(SOFA_COMPONENT_COLLISION_TETRAHEDRONBARYCENTRICCONTACTMAPPER_CPP)
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API ContactMapper<collision::geometry::TetrahedronCollisionModel, sofa::defaulttype::Vec3Types>;

#  ifdef _MSC_VER
// Manual declaration of non-specialized members, to avoid warnings from MSVC.
extern template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API void BarycentricContactMapper<collision::geometry::TetrahedronCollisionModel, defaulttype::Vec3Types>::cleanup();
extern template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API core::behavior::MechanicalState<defaulttype::Vec3Types>* BarycentricContactMapper<collision::geometry::TetrahedronCollisionModel, defaulttype::Vec3Types>::createMapping(const char*);
#  endif
#endif

} // namespace sofa::component::collision::response::mapper
