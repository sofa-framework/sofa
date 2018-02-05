/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToLineConstraint_CPP
#include <SofaBoundaryCondition/ProjectToLineConstraint.inl>
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


SOFA_DECL_CLASS(ProjectToLineConstraint)

int ProjectToLineConstraintClass = core::RegisterObject("Attach given particles to their initial positions")
#ifndef SOFA_FLOAT
        .add< ProjectToLineConstraint<Vec3dTypes> >()
        .add< ProjectToLineConstraint<Vec2dTypes> >()
//.add< ProjectToLineConstraint<Vec1dTypes> >()
//.add< ProjectToLineConstraint<Vec6dTypes> >()
//.add< ProjectToLineConstraint<Rigid3dTypes> >()
//.add< ProjectToLineConstraint<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ProjectToLineConstraint<Vec3fTypes> >()
        .add< ProjectToLineConstraint<Vec2fTypes> >()
//.add< ProjectToLineConstraint<Vec1fTypes> >()
//.add< ProjectToLineConstraint<Vec6fTypes> >()
//.add< ProjectToLineConstraint<Rigid3fTypes> >()
//.add< ProjectToLineConstraint<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Vec2dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Vec1dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Vec6dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Rigid3dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Vec2fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Vec1fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Vec6fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Rigid3fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToLineConstraint<Rigid2fTypes>;
#endif



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

