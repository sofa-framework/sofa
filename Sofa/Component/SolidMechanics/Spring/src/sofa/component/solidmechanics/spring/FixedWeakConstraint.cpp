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
#define SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGSFORCEFIELD_CPP

#include <sofa/component/solidmechanics/spring/FixedWeakConstraint.inl>

#include <sofa/helper/visual/DrawTool.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/component/solidmechanics/spring/BaseRestShapeSpringsForceField.inl>


namespace sofa::component::solidmechanics::spring
{

using namespace sofa::type;
using namespace sofa::defaulttype;

int FixedWeakConstraintClass = core::RegisterObject("Weak constraints fixing dofs at their rest shape using springs")
        .add< FixedWeakConstraint<Vec6Types> >()
        .add< FixedWeakConstraint<Vec3Types> >()
        .add< FixedWeakConstraint<Vec2Types> >()
        .add< FixedWeakConstraint<Vec1Types> >()
        .add< FixedWeakConstraint<Rigid3Types> >()

        ;

template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FixedWeakConstraint<Vec6Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FixedWeakConstraint<Vec3Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FixedWeakConstraint<Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FixedWeakConstraint<Vec1Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FixedWeakConstraint<Rigid3Types>;


} // namespace sofa::component::solidmechanics::spring
