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
#define SOFABOUNDARYCONDITION_AFFINEMOVEMENT_CONSTRAINT_CPP

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/simulation/Node.h>

#include <SofaBoundaryCondition/AffineMovementConstraint.inl>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

int AffineMovementConstraintClass = core::RegisterObject("bilinear constraint")
        .add< AffineMovementConstraint<defaulttype::Vec3Types> >()
        .add< AffineMovementConstraint<defaulttype::Rigid3Types> >();

template class SOFA_BOUNDARY_CONDITION_API AffineMovementConstraint<defaulttype::Vec3Types>;
template class SOFA_BOUNDARY_CONDITION_API AffineMovementConstraint<defaulttype::Rigid3Types>;

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa
