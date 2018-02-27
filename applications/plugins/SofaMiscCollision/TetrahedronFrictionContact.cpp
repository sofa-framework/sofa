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
#include <SofaConstraint/FrictionContact.inl>
#include <SofaMiscCollision/TetrahedronModel.h>

using namespace sofa::core::collision;

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(TetrahedronFrictionContact)

Creator<Contact::Factory, FrictionContact<TetrahedronModel, SphereModel> > TetrahedronSphereFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, PointModel> > TetrahedronPointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, LineModel> > TetrahedronLineFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, TriangleModel> > TetrahedronTriangleFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, TetrahedronModel> > TetrahedronTetrahedronFrictionContactClass("FrictionContact",true);

Creator<Contact::Factory, FrictionContact<TetrahedronModel, SphereModel> > TetrahedronSpherePenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, PointModel> > TetrahedronPointPenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, LineModel> > TetrahedronLinePenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, TriangleModel> > TetrahedronTrianglePenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, TetrahedronModel> > TetrahedronTetrahedronPenalityFrictionContactClass("FrictionContact",true);

} // namespace collision

} // namespace component

} // namespace sofa
