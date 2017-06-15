/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

Creator<Contact::Factory, DisabledContact<SphereModel, SphereModel> > SphereSphereDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<SphereModel, PointModel> > SpherePointDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<PointModel, PointModel> > PointPointDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<LineModel, PointModel> > LinePointDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<LineModel, LineModel> > LineLineDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<LineModel, SphereModel> > LineSphereDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<TriangleModel, SphereModel> > TriangleSphereDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<TriangleModel, PointModel> > TrianglePointDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<TriangleModel, LineModel> > TriangleLineDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<TriangleModel, TriangleModel> > TriangleTriangleDisabledContactClass("disabled",true);


Creator<Contact::Factory, DisabledContact<TriangleModel, RigidSphereModel> > TriangleRigidSphereDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<CapsuleModel, TriangleModel> > CapsuleTriangleDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<CapsuleModel, LineModel> > CapsuleLineDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<CapsuleModel, CapsuleModel> > CapsuleCapsuleDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<CapsuleModel, SphereModel> > CapsuleSphereDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<OBBModel, OBBModel> > OBBOBBDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<CapsuleModel, OBBModel> > CapsuleOBBDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<SphereModel, OBBModel> > SphereOBBDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<RigidSphereModel, OBBModel> > RigidSphereOBBDisabledContactClass("disabled",true);
Creator<Contact::Factory, DisabledContact<TriangleModel, OBBModel> > TriangleOBBDisabledContactClass("disabled",true);

} // namespace collision

} // namespace component

} // namespace sofa
