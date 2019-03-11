/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define FRAME_FRAMEFIXEDCONSTRAINT_CPP

#include "QuadraticTypes.h"
#include "AffineTypes.h"
#include "FrameFixedConstraint.inl"
#include <SofaBoundaryCondition/FixedConstraint.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;

int FrameFixedConstraintClass = core::RegisterObject("Cancel some degrees of freedom in the frames")
        .add< FrameFixedConstraint<Rigid3Types> >()
        .add< FrameFixedConstraint<Affine3dTypes> >()
        .add< FrameFixedConstraint<Quadratic3dTypes> >()

        ;

template class SOFA_FRAME_API FrameFixedConstraint<Rigid3Types>;
template class SOFA_FRAME_API FrameFixedConstraint<Affine3dTypes>;
template class SOFA_FRAME_API FrameFixedConstraint<Quadratic3dTypes>;



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa
