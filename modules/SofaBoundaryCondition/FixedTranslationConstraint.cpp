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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDTRANSLATIONCONSTRAINT_CPP
#include <SofaBoundaryCondition/FixedTranslationConstraint.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

int FixedTranslationConstraintClass = core::RegisterObject("Attach given rigids to their initial positions but they still can have rotations")
        .add< FixedTranslationConstraint<Rigid3Types> >()
        .add< FixedTranslationConstraint<Rigid2Types> >()
        .add< FixedTranslationConstraint<Vec6Types> >()

        ;

template class SOFA_BOUNDARY_CONDITION_API FixedTranslationConstraint<Rigid3Types>;
template class SOFA_BOUNDARY_CONDITION_API FixedTranslationConstraint<Rigid2Types>;
template class SOFA_BOUNDARY_CONDITION_API FixedTranslationConstraint<Vec6Types>;


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

