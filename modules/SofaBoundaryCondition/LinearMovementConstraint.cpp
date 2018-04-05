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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARMOVEMENTCONSTRAINT_CPP
#include <SofaBoundaryCondition/LinearMovementConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

//declaration of the class, for the factory
SOFA_DECL_CLASS(LinearMovementConstraint)


int LinearMovementConstraintClass = core::RegisterObject("translate given particles")
#ifndef SOFA_FLOAT
        .add< LinearMovementConstraint<defaulttype::Vec3dTypes> >()
        .add< LinearMovementConstraint<defaulttype::Vec2dTypes> >()
        .add< LinearMovementConstraint<defaulttype::Vec1dTypes> >()
        .add< LinearMovementConstraint<defaulttype::Vec6dTypes> >()
        .add< LinearMovementConstraint<defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LinearMovementConstraint<defaulttype::Vec3fTypes> >()
        .add< LinearMovementConstraint<defaulttype::Vec2fTypes> >()
        .add< LinearMovementConstraint<defaulttype::Vec1fTypes> >()
        .add< LinearMovementConstraint<defaulttype::Vec6fTypes> >()
        .add< LinearMovementConstraint<defaulttype::Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec1dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec6dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec1fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec6fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Rigid3fTypes>;
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa
