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
#include <SofaConstraint/FrictionContact.inl>

#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaMeshCollision/BarycentricContactMapper.inl>

namespace sofa::component::collision
{

using namespace defaulttype;
using namespace sofa::helper;
using simulation::Node;

Creator<sofa::core::collision::Contact::Factory, FrictionContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > LineSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLineFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleLineFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleTriangleFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > SpherePointFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > SphereRigidSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > LineRigidSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > TriangleRigidSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > RigidSpherePointFrictionContactClass("FrictionContact",true);


} //namespace sofa::component::collision
