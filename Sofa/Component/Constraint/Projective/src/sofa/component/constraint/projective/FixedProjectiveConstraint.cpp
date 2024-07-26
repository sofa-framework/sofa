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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPROJECTIVECONSTRAINT_CPP
#include <sofa/component/constraint/projective/FixedProjectiveConstraint.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Node.h>

namespace sofa::component::constraint::projective
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


int FixedProjectiveConstraintClass = core::RegisterObject("Attach given particles to their initial positions")
        .add< FixedProjectiveConstraint<Vec3Types> >()
        .add< FixedProjectiveConstraint<Vec2Types> >()
        .add< FixedProjectiveConstraint<Vec1Types> >()
        .add< FixedProjectiveConstraint<Vec6Types> >()
        .add< FixedProjectiveConstraint<Rigid3Types> >()
        .add< FixedProjectiveConstraint<Rigid2Types> >()
        ;

template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API FixedProjectiveConstraint<Vec3Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API FixedProjectiveConstraint<Vec2Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API FixedProjectiveConstraint<Vec1Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API FixedProjectiveConstraint<Vec6Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API FixedProjectiveConstraint<Rigid3Types>;
template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API FixedProjectiveConstraint<Rigid2Types>;

} // namespace sofa::component::constraint::projective
