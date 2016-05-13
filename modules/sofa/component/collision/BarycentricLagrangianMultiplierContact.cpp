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
#include <sofa/component/collision/BarycentricLagrangianMultiplierContact.inl>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace collision;

SOFA_DECL_CLASS(BarycentricLagrangianMultiplierContact)

Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<PointModel, PointModel> > PointPointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<LineModel, PointModel> > LinePointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<LineModel, LineModel> > LineLineLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleModel, PointModel> > TrianglePointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleModel, LineModel> > TriangleLineLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleModel, TriangleModel> > TriangleTriangleLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereModel, SphereModel> > SphereSphereLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereModel, PointModel> > SpherePointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<LineModel, SphereModel> > LineSphereLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleModel, SphereModel> > TriangleSphereLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereTreeModel,SphereTreeModel> > SphereTreeSphereTreeLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereTreeModel,TriangleModel> > SphereTreeTriangleLagrangianMultiplierContactClass("LagrangianMultiplier",true);

} // namespace collision

} // namespace component

} // namespace sofa

