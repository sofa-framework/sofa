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
//
// C++ Implementation: CurveMapping
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#define SOFA_COMPONENT_MAPPING_CURVEMAPPING_CPP

#include <SofaMiscMapping/CurveMapping.inl>

#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(CurveMapping)

// Register in the Factory
int CurveMappingClass = core::RegisterObject("Mapping allowing one or more rigid objects follow a trajectory determined by a set of points")

#ifndef SOFA_FLOAT
        .add< CurveMapping< Vec3dTypes, Rigid3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< CurveMapping< Vec3fTypes, Rigid3fTypes > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< CurveMapping< Vec3dTypes, Rigid3fTypes > >()
        .add< CurveMapping< Vec3fTypes, Rigid3dTypes > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_MAPPING_API CurveMapping< Vec3dTypes, Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API CurveMapping< Vec3fTypes, Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API CurveMapping< Vec3dTypes, Rigid3fTypes >;
template class SOFA_MISC_MAPPING_API CurveMapping< Vec3fTypes, Rigid3dTypes >;
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa
