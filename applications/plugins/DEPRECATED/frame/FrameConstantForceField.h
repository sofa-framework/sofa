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
#ifndef FRAME_FRAMECONSTANTFROCEFIELD_H
#define FRAME_FRAMECONSTANTFROCEFIELD_H

#include <SofaBoundaryCondition/ConstantForceField.h>
#include "AffineTypes.h"
#include "DeformationGradientTypes.h"
#include "initFrame.h"

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FRAME_FRAMECONSTANTFROCEFIELD_CPP)
#ifndef SOFA_FLOAT
//                extern template class SOFA_FRAME_API ConstantForceField<DeformationGradient331dTypes>;
extern template class SOFA_FRAME_API ConstantForceField<Affine3dTypes>;
extern template class SOFA_FRAME_API ConstantForceField<Quadratic3dTypes>;
#endif
#ifndef SOFA_DOUBLE
//                extern template class SOFA_FRAME_API ConstantForceField<DeformationGradient331fTypes>;
extern template class SOFA_FRAME_API ConstantForceField<Affine3fTypes>;
extern template class SOFA_FRAME_API ConstantForceField<Quadratic3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace core

} // namespace sofa

#endif
