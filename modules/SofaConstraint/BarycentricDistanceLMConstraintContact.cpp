/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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

sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<SphereModel, SphereModel> > SphereSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<SphereModel, PointModel> > SpherePointDistanceLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<PointModel, PointModel> > PointPointDistanceLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<LineModel, PointModel> > LinePointDistanceLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<LineModel, LineModel> > LineLineDistanceLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<LineModel, SphereModel> > LineSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<TriangleModel, SphereModel> > TriangleSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<TriangleModel, PointModel> > TrianglePointDistanceLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<TriangleModel, LineModel> > TriangleLineDistanceLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<TriangleModel, TriangleModel> > TriangleTriangleDistanceLMConstraintContactClass("distanceLMConstraint",true);

sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<TriangleModel, RigidSphereModel> > TriangleRigidSphereLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<TriangleModel, PointModel> > TrianglePointLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<TriangleModel, LineModel> > TriangleLineLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<TriangleModel, TriangleModel> > TriangleTriangleLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CapsuleModel, TriangleModel> > CapsuleTriangleLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CapsuleModel, LineModel> > CapsuleLineLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CapsuleModel, CapsuleModel> > CapsuleCapsuleLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CapsuleModel, SphereModel> > CapsuleSphereLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<OBBModel, OBBModel> > OBBOBBLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CapsuleModel, OBBModel> > CapsuleOBBLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<SphereModel, OBBModel> > SphereOBBLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<RigidSphereModel, OBBModel> > RigidSphereOBBLMConstraintContactClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<TriangleModel, OBBModel> > TriangleOBBLMConstraintContactClass("distanceLMConstraint",true);


sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<RigidCapsuleModel, TriangleModel> > RigidCapsuleTriangleLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<RigidCapsuleModel, RigidCapsuleModel> > RigidCapsuleRigidCapsuleLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CapsuleModel, RigidCapsuleModel> > CapsuleRigidCapsuleLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<RigidCapsuleModel, SphereModel> > RigidCapsuleSphereLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<RigidCapsuleModel, RigidSphereModel> > RigidCapsuleRigidSphereLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<RigidCapsuleModel, OBBModel> > RigidCapsuleOBBLMConstraintContactClassClass("distanceLMConstraint",true);


sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CylinderModel, CylinderModel> > CylinderCylinderLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CylinderModel, TriangleModel> > CylinderTriangleLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CylinderModel, RigidCapsuleModel> > CylinderRigidCapsuleLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CapsuleModel, CylinderModel> > CapsuleCylinderLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CylinderModel, SphereModel> > CylinderSphereLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CylinderModel, RigidSphereModel> > CylinderRigidSphereLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<CylinderModel, OBBModel> > CylinderOBBLMConstraintContactClassClass("distanceLMConstraint",true);

} // namespace collision

} // namespace component

} // namespace sofa

