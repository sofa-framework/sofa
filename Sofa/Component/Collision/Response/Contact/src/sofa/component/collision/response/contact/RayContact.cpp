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
#include <sofa/component/collision/response/contact/RayContact.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/geometry/RayCollisionModel.h>
#include <sofa/component/collision/geometry/SphereCollisionModel.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>

namespace sofa::component::collision::response::contact
{

using namespace sofa::defaulttype;
using namespace sofa::component::collision::geometry;

Creator<core::collision::Contact::Factory, RayContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>> > RaySphereContactClass("RayContact",true);
Creator<core::collision::Contact::Factory, RayContact<RigidSphereModel> > RayRigidSphereContactClass("RayContact",true);
Creator<core::collision::Contact::Factory, RayContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > RayTriangleContactClass("RayContact",true);

template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API RayContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API RayContact<RigidSphereModel>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API RayContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;

BaseRayContact::BaseRayContact(CollisionModel1* model1, core::collision::Intersection* /*instersectionMethod*/)
    : model1(model1)
{
    if (model1!=nullptr)
        model1->addContact(this);
}

BaseRayContact::~BaseRayContact()
{
    if (model1!=nullptr)
        model1->removeContact(this);
}


} //namespace sofa::component::collision::response::contact
