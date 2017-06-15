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
// C++ Implementation: TriangleBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_TRIANGLEBENDINGSPRINGS_CPP
#include <SofaGeneralDeformable/TriangleBendingSprings.inl>
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



SOFA_DECL_CLASS(TriangleBendingSprings)

// Register in the Factory
int TriangleBendingSpringsClass = core::RegisterObject("Springs added to a traingular mesh to prevent bending")
#ifndef SOFA_FLOAT
        .add< TriangleBendingSprings<Vec3dTypes> >()
        .add< TriangleBendingSprings<Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TriangleBendingSprings<Vec3fTypes> >()
        .add< TriangleBendingSprings<Vec2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_DEFORMABLE_API TriangleBendingSprings<Vec3dTypes>;
template class SOFA_GENERAL_DEFORMABLE_API TriangleBendingSprings<Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_DEFORMABLE_API TriangleBendingSprings<Vec3fTypes>;
template class SOFA_GENERAL_DEFORMABLE_API TriangleBendingSprings<Vec2fTypes>;
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

