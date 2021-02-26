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
#include <SofaMeshCollision/RigidContactMapper.inl>
#include <sofa/helper/Factory.inl>

namespace sofa::component::collision
{

using namespace defaulttype;

ContactMapperCreator< ContactMapper<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>,Vec3Types> > CylinderModelContactMapperClass("penality", true);
ContactMapperCreator< ContactMapper<RigidSphereModel,Vec3Types> > RigidSphereContactMapperClass("penality", true);
ContactMapperCreator< ContactMapper<OBBCollisionModel<sofa::defaulttype::Rigid3Types>,Vec3Types> > OBBContactMapperClass("penality", true);
ContactMapperCreator< ContactMapper<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>,Vec3Types> > RigidCapsuleContactMapperClass("penality", true);


template class SOFA_SOFAMESHCOLLISION_API ContactMapper<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>,Vec3Types>;
template class SOFA_SOFAMESHCOLLISION_API ContactMapper<RigidSphereModel,Vec3Types>;
template class SOFA_SOFAMESHCOLLISION_API ContactMapper<OBBCollisionModel<sofa::defaulttype::Rigid3Types>,Vec3Types>;
template class SOFA_SOFAMESHCOLLISION_API ContactMapper<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>,Vec3Types>;

template SOFA_SOFAMESHCOLLISION_API void RigidContactMapper<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>::cleanup();
template SOFA_SOFAMESHCOLLISION_API core::behavior::MechanicalState<defaulttype::Vec3Types>* RigidContactMapper<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>::createMapping(const char*);
template SOFA_SOFAMESHCOLLISION_API void RigidContactMapper<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>::cleanup();
template SOFA_SOFAMESHCOLLISION_API core::behavior::MechanicalState<defaulttype::Vec3Types>* RigidContactMapper<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>::createMapping(const char*);
template SOFA_SOFAMESHCOLLISION_API void RigidContactMapper<RigidSphereModel, defaulttype::Vec3Types>::cleanup();
template SOFA_SOFAMESHCOLLISION_API core::behavior::MechanicalState<defaulttype::Vec3Types>* RigidContactMapper<RigidSphereModel, defaulttype::Vec3Types>::createMapping(const char*);
template SOFA_SOFAMESHCOLLISION_API void RigidContactMapper<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>::cleanup();
template SOFA_SOFAMESHCOLLISION_API core::behavior::MechanicalState<defaulttype::Vec3Types>* RigidContactMapper<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>::createMapping(const char*);

} //namespace sofa::component::collision
