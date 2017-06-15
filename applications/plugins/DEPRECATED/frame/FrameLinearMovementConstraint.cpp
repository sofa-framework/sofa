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

SOFA_DECL_CLASS ( FrameLinearMovementConstraint )

using namespace sofa::defaulttype;

int LinearMovementConstraintClass = core::RegisterObject ( "mechanical state vectors" )
#ifndef SOFA_FLOAT
        .add< LinearMovementConstraint<Affine3dTypes> >()
        .add< LinearMovementConstraint<Quadratic3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LinearMovementConstraint<Affine3fTypes> >()
        .add< LinearMovementConstraint<Quadratic3fTypes> >()
#endif
        ;



#ifndef SOFA_FLOAT
template class SOFA_FRAME_API LinearMovementConstraint<Affine3dTypes>;
template class SOFA_FRAME_API LinearMovementConstraint<Quadratic3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API LinearMovementConstraint<Affine3fTypes>;
template class SOFA_FRAME_API LinearMovementConstraint<Quadratic3fTypes>;
#endif
} // namespace behavior

} // namespace core

} // namespace sofa
