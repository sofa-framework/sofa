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
#define SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_CPP
#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>
#include <sofa/helper/Factory.inl>

namespace sofa::component::collision::response::mapper
{

using namespace defaulttype;
using namespace sofa::component::collision::geometry;

ContactMapperCreator< ContactMapper<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>,Vec3Types> > CylinderModelContactMapperClass("PenalityContactForceField", true);
ContactMapperCreator< ContactMapper<RigidSphereModel,Vec3Types> > RigidSphereContactMapperClass("PenalityContactForceField", true);


template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API ContactMapper<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>,Vec3Types>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API ContactMapper<RigidSphereModel,Vec3Types>;

template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API void RigidContactMapper<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>::cleanup();
template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API core::behavior::MechanicalState<defaulttype::Vec3Types>* RigidContactMapper<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>::createMapping(const char*);
template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API void RigidContactMapper<RigidSphereModel, defaulttype::Vec3Types>::cleanup();
template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API core::behavior::MechanicalState<defaulttype::Vec3Types>* RigidContactMapper<RigidSphereModel, defaulttype::Vec3Types>::createMapping(const char*);

} //namespace sofa::component::collision::response::mapper
