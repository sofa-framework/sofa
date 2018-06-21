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
#define SOFA_COMPONENT_COLLISION_BARYCENTRICPENALITYCONTACT_CPP
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

Creator<Contact::Factory, BarycentricPenalityContact<SphereModel, SphereModel> > SphereSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereModel, RigidSphereModel> > SphereRigidSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereModel, PointModel> > SpherePointPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidSphereModel, PointModel> > RigidSpherePointPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<PointModel, PointModel> > PointPointPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineModel, PointModel> > LinePointPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineModel, LineModel> > LineLinePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineModel, SphereModel> > LineSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineModel, RigidSphereModel> > LineRigidSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleModel, SphereModel> > TriangleSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleModel, RigidSphereModel> > TriangleRigidSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleModel, PointModel> > TrianglePointPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleModel, LineModel> > TriangleLinePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleModel, TriangleModel> > TriangleTrianglePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleModel, TriangleModel> > CapsuleTrianglePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleModel, LineModel> > CapsuleLinePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleModel, CapsuleModel> > CapsuleCapsulePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleModel, SphereModel> > CapsuleSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<OBBModel, OBBModel> > OBBOBBPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleModel, OBBModel> > CapsuleOBBPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereModel, OBBModel> > SphereOBBPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidSphereModel, OBBModel> > RigidSphereOBBPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleModel, OBBModel> > TriangleOBBPenalityContactClass("default",true);

Creator<Contact::Factory, BarycentricPenalityContact<RigidCapsuleModel, TriangleModel> > RigidCapsuleTrianglePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidCapsuleModel, LineModel> > RigidCapsuleLinePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidCapsuleModel, RigidCapsuleModel> > RigidCapsuleRigidCapsulePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleModel, RigidCapsuleModel> > CapsuleRigidCapsulePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidCapsuleModel, SphereModel> > RigidCapsuleSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidCapsuleModel, RigidSphereModel> > RigidCapsuleRigidSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidCapsuleModel, OBBModel> > RigidCapsuleOBBPenalityContactClass("default",true);

Creator<Contact::Factory, BarycentricPenalityContact<CylinderModel, CylinderModel> > CylinderCylinderPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderModel, TriangleModel> > CylinderTrianglePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderModel, RigidCapsuleModel> > CylinderRigidCapsulePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleModel, CylinderModel> > CapsuleCylinderPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderModel, SphereModel> > CylinderSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderModel, RigidSphereModel> > CylinderRigidSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderModel, OBBModel> > CylinderOBBPenalityContactClass("default",true);



extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<SphereModel, SphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<SphereModel, RigidSphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<RigidSphereModel, RigidSphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<SphereModel, PointModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<RigidSphereModel, PointModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<PointModel, PointModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<LineModel, PointModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<LineModel, LineModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<LineModel, SphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<LineModel, RigidSphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<TriangleModel, SphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<TriangleModel, RigidSphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<TriangleModel, PointModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<TriangleModel, LineModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<TriangleModel, TriangleModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CapsuleModel, TriangleModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CapsuleModel, LineModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CapsuleModel, CapsuleModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CapsuleModel, SphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CapsuleModel, RigidSphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<OBBModel, OBBModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CapsuleModel, OBBModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<SphereModel, OBBModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<RigidSphereModel, OBBModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<TriangleModel, OBBModel>;

extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<RigidCapsuleModel, TriangleModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<RigidCapsuleModel, LineModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<RigidCapsuleModel, RigidCapsuleModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CapsuleModel, RigidCapsuleModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<RigidCapsuleModel, SphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<RigidCapsuleModel, RigidSphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<RigidCapsuleModel, OBBModel>;

extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CylinderModel, CylinderModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CylinderModel, TriangleModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CylinderModel, RigidCapsuleModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CapsuleModel, CylinderModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CylinderModel, SphereModel>; 
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CylinderModel, RigidSphereModel>;
extern template class SOFA_MESH_COLLISION_API BarycentricPenalityContact<CylinderModel, OBBModel>;

} // namespace collision

} // namespace component

} // namespace sofa

