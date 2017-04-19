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

sofa::core::collision::DetectionOutput::ContactId Identifier::cpt=0;
std::list<sofa::core::collision::DetectionOutput::ContactId> Identifier::availableId;

SOFA_DECL_CLASS(FrictionContact)

sofa::core::collision::ContactCreator< FrictionContact<PointModel, PointModel> > PointPointFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<LineModel, SphereModel> > LineSphereFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<LineModel, PointModel> > LinePointFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<LineModel, LineModel> > LineLineFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<TriangleModel, SphereModel> > TriangleSphereFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<TriangleModel, PointModel> > TrianglePointFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<TriangleModel, LineModel> > TriangleLineFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<TriangleModel, TriangleModel> > TriangleTriangleFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<SphereModel, SphereModel> > SphereSphereFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<SphereModel, PointModel> > SpherePointFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<CapsuleModel, CapsuleModel> > CapsuleCapsuleFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<CapsuleModel, TriangleModel> > CapsuleTriangleFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<CapsuleModel, SphereModel> > CapsuleSphereFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<OBBModel, OBBModel> > OBBOBBFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<SphereModel, OBBModel> > SphereOBBFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<CapsuleModel, OBBModel> > CapsuleOBBFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<TriangleModel, OBBModel> > TriangleOBBFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<SphereModel, RigidSphereModel> > SphereRigidSphereFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<LineModel, RigidSphereModel> > LineRigidSphereFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<TriangleModel, RigidSphereModel> > TriangleRigidSphereFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<RigidSphereModel, PointModel> > RigidSpherePointFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<RigidSphereModel, OBBModel> > RigidSphereOBBFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<RigidCapsuleModel, RigidCapsuleModel> > RigidCapsuleRigidCapsuleFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<CapsuleModel, RigidCapsuleModel> > CapsuleRigidCapsuleFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<RigidCapsuleModel, TriangleModel> > RigidCapsuleTriangleFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<RigidCapsuleModel, SphereModel> > RigidCapsuleSphereFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<RigidCapsuleModel, OBBModel> > RigidCapsuleOBBFrictionContactClass("FrictionContact",true);
sofa::core::collision::ContactCreator< FrictionContact<RigidCapsuleModel, RigidSphereModel> > RigidCapsuleRigidSphereFrictionContactClass("FrictionContact",true);


} // namespace collision

} // namespace component

} // namespace sofa
