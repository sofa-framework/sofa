/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/defaulttype/VecTypes.h>
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
int LinearVelocityConstraintClass = core::RegisterObject("apply velocity to given particles")
        .add< LinearVelocityConstraint<Vec3Types> >()
        .add< LinearVelocityConstraint<Vec2Types> >()
        .add< LinearVelocityConstraint<Vec1Types> >()
        .add< LinearVelocityConstraint<Vec6Types> >()
        .add< LinearVelocityConstraint<Rigid3Types> >()

        ;

template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec3Types>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec2Types>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec1Types>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Vec6Types>;
template class SOFA_BOUNDARY_CONDITION_API LinearVelocityConstraint<Rigid3Types>;


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

