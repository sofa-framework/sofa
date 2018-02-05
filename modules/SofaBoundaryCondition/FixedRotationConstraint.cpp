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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDROTATIONCONSTRAINT_CPP
#include <SofaBoundaryCondition/FixedRotationConstraint.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(FixedRotationConstraint)

int FixedRotationConstraintClass = core::RegisterObject("Prevents rotation around x or/and y or/and z axis")

#ifndef SOFA_FLOAT
        .add< FixedRotationConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< FixedRotationConstraint<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class FixedRotationConstraint<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class FixedRotationConstraint<Rigid3fTypes>;
#endif


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

