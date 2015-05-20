/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaMeshCollision/BarycentricPenalityContact.inl>
#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaMeshCollision/RigidContactMapper.h>
#include <SofaBaseCollision/RigidCapsuleModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace core::collision;
using simulation::Node;

SOFA_DECL_CLASS(BarycentricPenalityContact)

sofa::core::collision::ContactCreator< BarycentricPenalityContact<SphereModel, SphereModel> > SphereSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<SphereModel, RigidSphereModel> > SphereRigidSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<SphereModel, PointModel> > SpherePointPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidSphereModel, PointModel> > RigidSpherePointPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<PointModel, PointModel> > PointPointPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<LineModel, PointModel> > LinePointPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<LineModel, LineModel> > LineLinePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<LineModel, SphereModel> > LineSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<LineModel, RigidSphereModel> > LineRigidSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<TriangleModel, SphereModel> > TriangleSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<TriangleModel, RigidSphereModel> > TriangleRigidSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<TriangleModel, PointModel> > TrianglePointPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<TriangleModel, LineModel> > TriangleLinePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<TriangleModel, TriangleModel> > TriangleTrianglePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CapsuleModel, TriangleModel> > CapsuleTrianglePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CapsuleModel, LineModel> > CapsuleLinePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CapsuleModel, CapsuleModel> > CapsuleCapsulePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CapsuleModel, SphereModel> > CapsuleSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<OBBModel, OBBModel> > OBBOBBPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CapsuleModel, OBBModel> > CapsuleOBBPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<SphereModel, OBBModel> > SphereOBBPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidSphereModel, OBBModel> > RigidSphereOBBPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<TriangleModel, OBBModel> > TriangleOBBPenalityContactClass("default",true);

sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidCapsuleModel, TriangleModel> > RigidCapsuleTrianglePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidCapsuleModel, LineModel> > RigidCapsuleLinePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidCapsuleModel, RigidCapsuleModel> > RigidCapsuleRigidCapsulePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CapsuleModel, RigidCapsuleModel> > CapsuleRigidCapsulePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidCapsuleModel, SphereModel> > RigidCapsuleSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidCapsuleModel, RigidSphereModel> > RigidCapsuleRigidSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidCapsuleModel, OBBModel> > RigidCapsuleOBBPenalityContactClass("default",true);

sofa::core::collision::ContactCreator< BarycentricPenalityContact<CylinderModel, CylinderModel> > CylinderCylinderPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CylinderModel, TriangleModel> > CylinderTrianglePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CylinderModel, RigidCapsuleModel> > CylinderRigidCapsulePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CapsuleModel, CylinderModel> > CapsuleCylinderPenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CylinderModel, SphereModel> > CylinderSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CylinderModel, RigidSphereModel> > CylinderRigidSpherePenalityContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<CylinderModel, OBBModel> > CylinderOBBPenalityContactClass("default",true);
} // namespace collision

} // namespace component

} // namespace sofa

