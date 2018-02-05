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
#include "BarycentricDistanceLMConstraintContact.inl"
#include <SofaMeshCollision/BarycentricContactMapper.h>

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

namespace sofa
{

namespace component
{

namespace collision
{

using simulation::Node;

SOFA_DECL_CLASS(BarycentricDistanceLMConstraintContact)

Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<SphereModel, SphereModel> > SphereSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<SphereModel, PointModel> > SpherePointDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<PointModel, PointModel> > PointPointDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<LineModel, PointModel> > LinePointDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<LineModel, LineModel> > LineLineDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<LineModel, SphereModel> > LineSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TriangleModel, SphereModel> > TriangleSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TriangleModel, PointModel> > TrianglePointDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TriangleModel, LineModel> > TriangleLineDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TriangleModel, TriangleModel> > TriangleTriangleDistanceLMConstraintContactClass("distanceLMConstraint",true);

Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TriangleModel, RigidSphereModel> > TriangleRigidSphereLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TriangleModel, PointModel> > TrianglePointLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TriangleModel, LineModel> > TriangleLineLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TriangleModel, TriangleModel> > TriangleTriangleLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CapsuleModel, TriangleModel> > CapsuleTriangleLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CapsuleModel, LineModel> > CapsuleLineLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CapsuleModel, CapsuleModel> > CapsuleCapsuleLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CapsuleModel, SphereModel> > CapsuleSphereLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<OBBModel, OBBModel> > OBBOBBLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CapsuleModel, OBBModel> > CapsuleOBBLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<SphereModel, OBBModel> > SphereOBBLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidSphereModel, OBBModel> > RigidSphereOBBLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TriangleModel, OBBModel> > TriangleOBBLMConstraintContactClass("distanceLMConstraint",true);


Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidCapsuleModel, TriangleModel> > RigidCapsuleTriangleLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidCapsuleModel, RigidCapsuleModel> > RigidCapsuleRigidCapsuleLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CapsuleModel, RigidCapsuleModel> > CapsuleRigidCapsuleLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidCapsuleModel, SphereModel> > RigidCapsuleSphereLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidCapsuleModel, RigidSphereModel> > RigidCapsuleRigidSphereLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidCapsuleModel, OBBModel> > RigidCapsuleOBBLMConstraintContactClassClass("distanceLMConstraint",true);


Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CylinderModel, CylinderModel> > CylinderCylinderLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CylinderModel, TriangleModel> > CylinderTriangleLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CylinderModel, RigidCapsuleModel> > CylinderRigidCapsuleLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CapsuleModel, CylinderModel> > CapsuleCylinderLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CylinderModel, SphereModel> > CylinderSphereLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CylinderModel, RigidSphereModel> > CylinderRigidSphereLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<CylinderModel, OBBModel> > CylinderOBBLMConstraintContactClassClass("distanceLMConstraint",true);

} // namespace collision

} // namespace component

} // namespace sofa

