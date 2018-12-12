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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_CPP
#include <SofaBoundaryCondition/FixedPlaneConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


int FixedPlaneConstraintClass = core::RegisterObject("Project particles on a given plane")
#ifdef SOFA_WITH_FLOAT
        .add< FixedPlaneConstraint<Vec3fTypes> >()
        .add< FixedPlaneConstraint<Vec6fTypes> >()
        .add< FixedPlaneConstraint<Rigid3fTypes> >()
#endif /// SOFA_WITH_FLOAT
#ifdef SOFA_WITH_DOUBLE
        .add< FixedPlaneConstraint<Rigid3dTypes> >()
        .add< FixedPlaneConstraint<Vec3dTypes> >()
        .add< FixedPlaneConstraint<Vec6dTypes> >()
#endif /// SOFA_WITH_DOUBLE
        ;

#ifdef SOFA_WITH_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Rigid3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec6dTypes>;
#endif /// SOFA_WITH_DOUBLE
#ifdef SOFA_WITH_FLOAT
template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Rigid3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec6fTypes>;
#endif /// SOFA_WITH_FLOAT

} /// namespace projectiveconstraintset

} /// namespace component

} /// namespace sofa

