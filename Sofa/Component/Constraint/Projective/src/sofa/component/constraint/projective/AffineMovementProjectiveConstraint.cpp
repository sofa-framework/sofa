/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#define SOFABOUNDARYCONDITION_AFFINEMOVEMENTPROJECTIVECONSTRAINT_CPP

#include <sofa/component/constraint/projective/config.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/component/constraint/projective/AffineMovementProjectiveConstraint.inl>

namespace sofa::component::constraint::projective
{

int AffineMovementProjectiveConstraintRegister = core::RegisterObject("Constraint the movement by a rigid transform.")
        .add< AffineMovementProjectiveConstraint<defaulttype::Vec3Types> >()
        .add< AffineMovementProjectiveConstraint<defaulttype::Rigid3Types> >()
        ;

template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AffineMovementProjectiveConstraint<defaulttype::Vec3Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AffineMovementProjectiveConstraint<defaulttype::Rigid3Types>;
 
} // namespace sofa::component::constraint::projective
