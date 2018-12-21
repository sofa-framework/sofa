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


int ExternalInterpolationMappingClass = core::RegisterObject("TODO-ExternalInterpolationMappingClass")
        .add< ExternalInterpolationMapping< Vec3dTypes, Vec3dTypes > >()
        .add< ExternalInterpolationMapping< Vec2Types, Vec2Types > >()
        .add< ExternalInterpolationMapping< Vec1Types, Vec1Types > >()
        .add< ExternalInterpolationMapping< Vec2Types, ExtVec2Types > >()
        .add< ExternalInterpolationMapping< Vec3dTypes, ExtVec3Types > >()



        ;


template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec2Types, Vec2Types >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec1Types, Vec1Types >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec3dTypes, ExtVec3Types >;
template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< Vec2Types, ExtVec2Types >;




// Mech -> Mech

// Mech -> Mapped

// Mech -> ExtMapped

} // namespace mapping

} // namespace component

} // namespace sofa
