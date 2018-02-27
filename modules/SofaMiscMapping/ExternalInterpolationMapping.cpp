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
#define SOFA_COMPONENT_MAPPING_EXTERNALINTERPOLATIONMAPPING_CPP

#include "ExternalInterpolationMapping.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(ExternalInterpolationMapping)

int ExternalInterpolationMappingClass = core::RegisterObject("TODO-ExternalInterpolationMappingClass")
#ifndef SOFA_FLOAT
        .add< ExternalInterpolationMapping< Vec3dTypes, Vec3dTypes > >()
        .add< ExternalInterpolationMapping< Vec2dTypes, Vec2dTypes > >()
        .add< ExternalInterpolationMapping< Vec1dTypes, Vec1dTypes > >()
        .add< ExternalInterpolationMapping< Vec2dTypes, ExtVec2fTypes > >()
        .add< ExternalInterpolationMapping< Vec3dTypes, ExtVec3fTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< ExternalInterpolationMapping< Vec3fTypes, Vec3fTypes > >()
        .add< ExternalInterpolationMapping< Vec2fTypes, Vec2fTypes > >()
        .add< ExternalInterpolationMapping< Vec1fTypes, Vec1fTypes > >()
        .add< ExternalInterpolationMapping< Vec2fTypes, ExtVec2fTypes > >()
        .add< ExternalInterpolationMapping< Vec3fTypes, ExtVec3fTypes > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< ExternalInterpolationMapping< Vec3fTypes, Vec3dTypes > >()
        .add< ExternalInterpolationMapping< Vec3dTypes, Vec3fTypes > >()
        .add< ExternalInterpolationMapping< Vec2fTypes, Vec2dTypes > >()
        .add< ExternalInterpolationMapping< Vec2dTypes, Vec2fTypes > >()
        .add< ExternalInterpolationMapping< Vec1fTypes, Vec1dTypes > >()
        .add< ExternalInterpolationMapping< Vec1dTypes, Vec1fTypes > >()
#endif
#endif
        ;


#ifndef SOFA_FLOAT
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec2dTypes, Vec2dTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec1dTypes, Vec1dTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec2dTypes, ExtVec2fTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec2fTypes, Vec2fTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec1fTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec2fTypes, ExtVec2fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec3dTypes, Vec3fTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec3fTypes, Vec3dTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec2dTypes, Vec2fTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec2fTypes, Vec2dTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec1dTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec1fTypes, Vec1dTypes >;
#endif
#endif

// Mech -> Mech

// Mech -> Mapped

// Mech -> ExtMapped

} // namespace mapping

} // namespace component

} // namespace sofa
