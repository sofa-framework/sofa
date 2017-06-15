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
//
// C++ Implementation: LineBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_LINEBENDINGSPRINGS_CPP
#include <SofaMiscForceField/LineBendingSprings.inl>
#include <SofaDeformable/StiffSpringForceField.inl>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;



SOFA_DECL_CLASS(LineBendingSprings)

// Register in the Factory
int LineBendingSpringsClass = core::RegisterObject("Springs added to a polyline to prevent bending")
#ifndef SOFA_FLOAT
        .add< LineBendingSprings<Vec3dTypes> >()
        .add< LineBendingSprings<Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LineBendingSprings<Vec3fTypes> >()
        .add< LineBendingSprings<Vec2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_FORCEFIELD_API LineBendingSprings<Vec3dTypes>;
template class SOFA_MISC_FORCEFIELD_API LineBendingSprings<Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MISC_FORCEFIELD_API LineBendingSprings<Vec3fTypes>;
template class SOFA_MISC_FORCEFIELD_API LineBendingSprings<Vec2fTypes>;
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

