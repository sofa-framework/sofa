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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARVELOCITYCONSTRAINT_CPP
#include <SofaBoundaryCondition/LinearVelocityConstraint.inl>
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

using namespace sofa::defaulttype;
using namespace sofa::helper;

//declaration of the class, for the factory
SOFA_DECL_CLASS(LinearVelocityConstraint)


int LinearVelocityConstraintClass = core::RegisterObject("apply velocity to given particles")
#ifndef SOFA_FLOAT
        .add< LinearVelocityConstraint<Vec3dTypes> >()
        .add< LinearVelocityConstraint<Vec2dTypes> >()
        .add< LinearVelocityConstraint<Vec1dTypes> >()
        .add< LinearVelocityConstraint<Vec6dTypes> >()
        .add< LinearVelocityConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LinearVelocityConstraint<Vec3fTypes> >()
        .add< LinearVelocityConstraint<Vec2fTypes> >()
        .add< LinearVelocityConstraint<Vec1fTypes> >()
        .add< LinearVelocityConstraint<Vec6fTypes> >()
        .add< LinearVelocityConstraint<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec1dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec6dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec1fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec6fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Rigid3fTypes>;
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

