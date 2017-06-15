/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POINTCONSTRAINT_CPP
#include <SofaBoundaryCondition/PointConstraint.inl>
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


SOFA_DECL_CLASS(PointConstraint)

int PointConstraintClass = core::RegisterObject("Attach given particles to their initial positions")
#ifndef SOFA_FLOAT
        .add< PointConstraint<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< PointConstraint<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API PointConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API PointConstraint<Vec3fTypes>;
#endif



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

