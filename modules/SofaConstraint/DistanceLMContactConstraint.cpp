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
#define SOFA_COMPONENT_CONSTRAINTSET_DistanceLMContactConstraint_CPP
#include <SofaConstraint/DistanceLMContactConstraint.inl>

#include <sofa/core/behavior/LMConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(DistanceLMContactConstraint)

int DistanceLMContactConstraintClass = core::RegisterObject("Maintain a minimum contact distance between two objects")
#ifndef SOFA_FLOAT
        .add< DistanceLMContactConstraint<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< DistanceLMContactConstraint<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_CONSTRAINT_API DistanceLMContactConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CONSTRAINT_API DistanceLMContactConstraint<Vec3fTypes>;
#endif




} // namespace constraintset

} // namespace component

} // namespace sofa

