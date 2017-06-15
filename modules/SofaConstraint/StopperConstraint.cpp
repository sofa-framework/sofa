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
#define SOFA_COMPONENT_CONSTRAINTSET_STOPPERCONSTRAINT_CPP
#include <SofaConstraint/StopperConstraint.inl>

#include <sofa/defaulttype/Vec3Types.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(StopperConstraint)

int StopperConstraintClass = core::RegisterObject("TODO-StopperConstraint")
#ifndef SOFA_FLOAT
        .add< StopperConstraint<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< StopperConstraint<Vec1fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class StopperConstraint<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class StopperConstraint<Vec1fTypes>;
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

