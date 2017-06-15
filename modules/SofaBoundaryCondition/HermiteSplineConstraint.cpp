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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_HERMITESPLINECONSTRAINT_CPP
#include <SofaBoundaryCondition/HermiteSplineConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

SOFA_DECL_CLASS(HermiteSplineConstraint)


int HermiteSplineConstraintClass = core::RegisterObject("Apply a hermite cubic spline trajectory to given points")
#ifndef SOFA_FLOAT
        .add< HermiteSplineConstraint<defaulttype::Vec3dTypes> >()
        .add< HermiteSplineConstraint<defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< HermiteSplineConstraint<defaulttype::Vec3fTypes> >()
        .add< HermiteSplineConstraint<defaulttype::Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class HermiteSplineConstraint<defaulttype::Rigid3dTypes>;
template class HermiteSplineConstraint<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class HermiteSplineConstraint<defaulttype::Rigid3fTypes>;
template class HermiteSplineConstraint<defaulttype::Vec3fTypes>;
#endif



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

