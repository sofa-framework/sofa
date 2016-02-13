/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPlaneUnilateralConstraint_CPP
#include <sofa/component/projectiveconstraintset/ProjectToPlaneUnilateralConstraint.inl>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


SOFA_DECL_CLASS(ProjectToPlaneUnilateralConstraint)

int ProjectToPlaneUnilateralConstraintClass = core::RegisterObject("Keep all the particles on the positive side of an affine plane")
#ifndef SOFA_FLOAT
        .add< ProjectToPlaneUnilateralConstraint<Vec3dTypes> >()
        .add< ProjectToPlaneUnilateralConstraint<Vec2dTypes> >()
//.add< ProjectToPlaneUnilateralConstraint<Vec1dTypes> >()
//.add< ProjectToPlaneUnilateralConstraint<Vec6dTypes> >()
//.add< ProjectToPlaneUnilateralConstraint<Rigid3dTypes> >()
//.add< ProjectToPlaneUnilateralConstraint<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ProjectToPlaneUnilateralConstraint<Vec3fTypes> >()
        .add< ProjectToPlaneUnilateralConstraint<Vec2fTypes> >()
//.add< ProjectToPlaneUnilateralConstraint<Vec1fTypes> >()
//.add< ProjectToPlaneUnilateralConstraint<Vec6fTypes> >()
//.add< ProjectToPlaneUnilateralConstraint<Rigid3fTypes> >()
//.add< ProjectToPlaneUnilateralConstraint<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Vec2dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Vec1dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Vec6dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Rigid3dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Vec2fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Vec1fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Vec6fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Rigid3fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneUnilateralConstraint<Rigid2fTypes>;
#endif


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

