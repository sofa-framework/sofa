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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARMOVEMENTPROJECTIVECONSTRAINT_CPP
#include <sofa/component/constraint/projective/LinearMovementProjectiveConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::constraint::projective
{

template <> SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API
void LinearMovementProjectiveConstraint<defaulttype::Rigid3Types>::init()
{    
    this->core::behavior::ProjectiveConstraintSet<defaulttype::Rigid3Types>::init();

    x0.resize(0);
    nextM = prevM = Deriv();

    currentTime = -1.0;
    finished = false;
}


//declaration of the class, for the factory
int LinearMovementProjectiveConstraintClass = core::RegisterObject("translate given particles")
        .add< LinearMovementProjectiveConstraint<defaulttype::Vec3Types> >()
        .add< LinearMovementProjectiveConstraint<defaulttype::Vec2Types> >()
        .add< LinearMovementProjectiveConstraint<defaulttype::Vec1Types> >()
        .add< LinearMovementProjectiveConstraint<defaulttype::Vec6Types> >()
        .add< LinearMovementProjectiveConstraint<defaulttype::Rigid3Types> >()
    ;


template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API LinearMovementProjectiveConstraint<defaulttype::Vec3Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API LinearMovementProjectiveConstraint<defaulttype::Vec2Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API LinearMovementProjectiveConstraint<defaulttype::Vec1Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API LinearMovementProjectiveConstraint<defaulttype::Vec6Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API LinearMovementProjectiveConstraint<defaulttype::Rigid3Types>;

} // namespace sofa::component::constraint::projective
