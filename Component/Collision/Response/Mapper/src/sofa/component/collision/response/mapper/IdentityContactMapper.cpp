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
#define SOFA_COMPONENT_COLLISION_IDENTITYCONTACTMAPPER_CPP
#include <sofa/component/collision/response/mapper/IdentityContactMapper.inl>

namespace sofa::component::collision::response::mapper
{

using namespace defaulttype;
using namespace sofa::component::collision::geometry;

ContactMapperCreator< ContactMapper<SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereContactMapperClass("PenalityContactForceField",true);
ContactMapperCreator< ContactMapper<PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointContactMapperClass("PenalityContactForceField",true);

template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API ContactMapper<SphereCollisionModel<sofa::defaulttype::Vec3Types>, sofa::defaulttype::Vec3Types>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API ContactMapper<PointCollisionModel<sofa::defaulttype::Vec3Types>, sofa::defaulttype::Vec3Types>;

} //namespace sofa::component::collision::response::mapper
