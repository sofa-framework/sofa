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
#include <SofaConstraint/StickContactConstraint.inl>
#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaMeshCollision/RigidContactMapper.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using namespace sofa::helper;
using simulation::Node;
using namespace sofa::core::collision;

SOFA_DECL_CLASS(StickContactConstraint)

sofa::core::collision::ContactCreator< StickContactConstraint<PointModel, PointModel> > PointPointStickContactConstraintClass("StickContactConstraint",true);
sofa::core::collision::ContactCreator< StickContactConstraint<LineModel, SphereModel> > LineSphereStickContactConstraintClass("StickContactConstraint",true);
sofa::core::collision::ContactCreator< StickContactConstraint<LineModel, PointModel> > LinePointStickContactConstraintClass("StickContactConstraint",true);
sofa::core::collision::ContactCreator< StickContactConstraint<LineModel, LineModel> > LineLineStickContactConstraintClass("StickContactConstraint",true);
sofa::core::collision::ContactCreator< StickContactConstraint<TriangleModel, SphereModel> > TriangleSphereStickContactConstraintClass("StickContactConstraint",true);
sofa::core::collision::ContactCreator< StickContactConstraint<TriangleModel, PointModel> > TrianglePointStickContactConstraintClass("StickContactConstraint",true);
sofa::core::collision::ContactCreator< StickContactConstraint<TriangleModel, LineModel> > TriangleLineStickContactConstraintClass("StickContactConstraint",true);
sofa::core::collision::ContactCreator< StickContactConstraint<TriangleModel, TriangleModel> > TriangleTriangleStickContactConstraintClass("StickContactConstraint",true);
sofa::core::collision::ContactCreator< StickContactConstraint<SphereModel, SphereModel> > SphereSphereStickContactConstraintClass("StickContactConstraint",true);
sofa::core::collision::ContactCreator< StickContactConstraint<SphereModel, PointModel> > SpherePointStickContactConstraintClass("StickContactConstraint",true);


//sofa::core::collision::ContactCreator< StickContactConstraint<TriangleModel, RigidSphereModel> > TriangleRigidSphereContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<TriangleModel, PointModel> > TrianglePointContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<TriangleModel, LineModel> > TriangleLineContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<TriangleModel, TriangleModel> > TriangleTriangleContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<CapsuleModel, TriangleModel> > CapsuleTriangleContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<CapsuleModel, LineModel> > CapsuleLineContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<CapsuleModel, CapsuleModel> > CapsuleCapsuleContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<CapsuleModel, SphereModel> > CapsuleSphereContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<OBBModel, OBBModel> > OBBOBBContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<CapsuleModel, OBBModel> > CapsuleOBBContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<SphereModel, OBBModel> > SphereOBBContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<RigidSphereModel, OBBModel> > RigidSphereOBBContactConstraintClass("StickContactConstraint",true);
//sofa::core::collision::ContactCreator< StickContactConstraint<TriangleModel, OBBModel> > TriangleOBBContactConstraintClass("StickContactConstraint",true);
} // namespace collision

} // namespace component

} // namespace sofa
