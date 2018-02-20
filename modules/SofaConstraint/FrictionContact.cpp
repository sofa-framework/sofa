/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using namespace sofa::helper;
using simulation::Node;

SOFA_DECL_CLASS(FrictionContact)

Creator<sofa::core::collision::Contact::Factory, FrictionContact<PointModel, PointModel> > PointPointFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineModel, SphereModel> > LineSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineModel, PointModel> > LinePointFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineModel, LineModel> > LineLineFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleModel, SphereModel> > TriangleSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleModel, PointModel> > TrianglePointFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleModel, LineModel> > TriangleLineFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleModel, TriangleModel> > TriangleTriangleFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereModel, SphereModel> > SphereSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereModel, PointModel> > SpherePointFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleModel, CapsuleModel> > CapsuleCapsuleFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleModel, TriangleModel> > CapsuleTriangleFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleModel, SphereModel> > CapsuleSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<OBBModel, OBBModel> > OBBOBBFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereModel, OBBModel> > SphereOBBFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleModel, OBBModel> > CapsuleOBBFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleModel, OBBModel> > TriangleOBBFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereModel, RigidSphereModel> > SphereRigidSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineModel, RigidSphereModel> > LineRigidSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleModel, RigidSphereModel> > TriangleRigidSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidSphereModel, PointModel> > RigidSpherePointFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidSphereModel, OBBModel> > RigidSphereOBBFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidCapsuleModel, RigidCapsuleModel> > RigidCapsuleRigidCapsuleFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleModel, RigidCapsuleModel> > CapsuleRigidCapsuleFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidCapsuleModel, TriangleModel> > RigidCapsuleTriangleFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidCapsuleModel, SphereModel> > RigidCapsuleSphereFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidCapsuleModel, OBBModel> > RigidCapsuleOBBFrictionContactClass("FrictionContact",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidCapsuleModel, RigidSphereModel> > RigidCapsuleRigidSphereFrictionContactClass("FrictionContact",true);


} // namespace collision

} // namespace component

} // namespace sofa
