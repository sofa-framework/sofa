/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/collision/DisabledContact.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/TriangleModel.h>

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


} // namespace collision

} // namespace component

} // namespace sofa
