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
#define FRAME_FRAMELINEARMOVEMENTCONSTRAINT_CPP

#include "FrameLinearMovementConstraint.h"
#include <SofaBoundaryCondition/LinearMovementConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/template.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;

int LinearMovementConstraintClass = core::RegisterObject ( "mechanical state vectors" )
        .add< LinearMovementConstraint<Affine3dTypes> >()
        .add< LinearMovementConstraint<Quadratic3dTypes> >()

        ;



template class SOFA_FRAME_API LinearMovementConstraint<Affine3dTypes>;
template class SOFA_FRAME_API LinearMovementConstraint<Quadratic3dTypes>;

} // namespace behavior

} // namespace core

} // namespace sofa
