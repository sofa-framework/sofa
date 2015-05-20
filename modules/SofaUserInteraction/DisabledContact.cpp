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
#include <SofaUserInteraction/DisabledContact.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseCollision/OBBModel.h>
#include <SofaBaseCollision/CapsuleModel.h>



namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(DisabledContact)


using namespace sofa::core::collision;

sofa::core::collision::ContactCreator< DisabledContact<SphereModel, SphereModel> > SphereSphereDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<SphereModel, PointModel> > SpherePointDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<PointModel, PointModel> > PointPointDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<LineModel, PointModel> > LinePointDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<LineModel, LineModel> > LineLineDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<LineModel, SphereModel> > LineSphereDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<TriangleModel, SphereModel> > TriangleSphereDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<TriangleModel, PointModel> > TrianglePointDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<TriangleModel, LineModel> > TriangleLineDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<TriangleModel, TriangleModel> > TriangleTriangleDisabledContactClass("disabled",true);


sofa::core::collision::ContactCreator< DisabledContact<TriangleModel, RigidSphereModel> > TriangleRigidSphereDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<CapsuleModel, TriangleModel> > CapsuleTriangleDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<CapsuleModel, LineModel> > CapsuleLineDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<CapsuleModel, CapsuleModel> > CapsuleCapsuleDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<CapsuleModel, SphereModel> > CapsuleSphereDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<OBBModel, OBBModel> > OBBOBBDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<CapsuleModel, OBBModel> > CapsuleOBBDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<SphereModel, OBBModel> > SphereOBBDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<RigidSphereModel, OBBModel> > RigidSphereOBBDisabledContactClass("disabled",true);
sofa::core::collision::ContactCreator< DisabledContact<TriangleModel, OBBModel> > TriangleOBBDisabledContactClass("disabled",true);

} // namespace collision

} // namespace component

} // namespace sofa
