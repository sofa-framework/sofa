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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPointConstraint_CPP
#include <SofaBoundaryCondition/ProjectToPointConstraint.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


SOFA_DECL_CLASS(ProjectToPointConstraint)

int ProjectToPointConstraintClass = core::RegisterObject("Project particles to a point")
#ifndef SOFA_FLOAT
        .add< ProjectToPointConstraint<Vec3dTypes> >()
        .add< ProjectToPointConstraint<Vec2dTypes> >()
        .add< ProjectToPointConstraint<Vec1dTypes> >()
        .add< ProjectToPointConstraint<Vec6dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ProjectToPointConstraint<Vec3fTypes> >()
        .add< ProjectToPointConstraint<Vec2fTypes> >()
        .add< ProjectToPointConstraint<Vec1fTypes> >()
        .add< ProjectToPointConstraint<Vec6fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<Vec1dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<Vec1fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToPointConstraint<Vec6fTypes>;
#endif


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

