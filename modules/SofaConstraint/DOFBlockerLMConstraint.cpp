/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#define SOFA_COMPONENT_CONSTRAINTSET_DOFBLOCKERLMCONSTRAINT_CPP
#include <SofaConstraint/DOFBlockerLMConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(DOFBlockerLMConstraint)

int DOFBlockerLMConstraintClass = core::RegisterObject("Constrain the rotation of a given set of Rigid Bodies")
#ifndef SOFA_FLOAT
        .add< DOFBlockerLMConstraint<Rigid3dTypes> >()
        .add< DOFBlockerLMConstraint<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< DOFBlockerLMConstraint<Rigid3fTypes> >()
        .add< DOFBlockerLMConstraint<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class DOFBlockerLMConstraint<Rigid3dTypes>;
template class DOFBlockerLMConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class DOFBlockerLMConstraint<Rigid3fTypes>;
template class DOFBlockerLMConstraint<Vec3fTypes>;
#endif



} // namespace constraintset

} // namespace component

} // namespace sofa

