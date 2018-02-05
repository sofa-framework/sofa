/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaMiscCollision/BarycentricStickContact.inl>
#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaMeshCollision/IdentityContactMapper.h>

using namespace sofa::core::collision ;

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(BarycentricStickContact)

Creator<Contact::Factory, BarycentricStickContact<SphereModel, SphereModel> > SphereSphereStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<SphereModel, PointModel> > SpherePointStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<PointModel, PointModel> > PointPointStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<LineModel, PointModel> > LinePointStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<LineModel, LineModel> > LineLineStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<LineModel, SphereModel> > LineSphereStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleModel, SphereModel> > TriangleSphereStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleModel, PointModel> > TrianglePointStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleModel, LineModel> > TriangleLineStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleModel, TriangleModel> > TriangleTriangleStickContactClass("stick",true);


} // namespace collision

} // namespace component

} // namespace sofa

