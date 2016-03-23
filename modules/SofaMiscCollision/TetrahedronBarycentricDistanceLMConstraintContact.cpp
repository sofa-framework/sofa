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
#include <SofaConstraint/BarycentricDistanceLMConstraintContact.inl>
#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaMiscCollision/TetrahedronModel.h>

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

namespace sofa
{

namespace component
{

namespace collision
{

Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TetrahedronModel, SphereModel> > TetrahedronSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TetrahedronModel, PointModel> > TetrahedronPointDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TetrahedronModel, LineModel> > TetrahedronLineDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TetrahedronModel, TriangleModel> > TetrahedronTriangleDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<TetrahedronModel, TetrahedronModel> > TetrahedronTetrahedronDistanceLMConstraintContactClass("distanceLMConstraint",true);

} // namespace collision

} // namespace component

} // namespace sofa

