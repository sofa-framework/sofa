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
#define SOFA_COMPONENT_MAPPING_ProjectionToPlaneMapping_CPP

#include "ProjectionToPlaneMapping.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int ProjectionToTargetPlaneMappingClass = core::RegisterObject("Compute distance between a moving point and fixed line")
        .add< ProjectionToTargetPlaneMapping< Vec3Types, Vec3Types > >()
        .add< ProjectionToTargetPlaneMapping< Rigid3Types, Vec3Types > >()

        ;

template class SOFA_MISC_MAPPING_API ProjectionToTargetPlaneMapping< Vec3Types, Vec3Types >;
template class SOFA_MISC_MAPPING_API ProjectionToTargetPlaneMapping< Rigid3Types, Vec3Types >;


///////////////////

using namespace defaulttype;

// Register in the Factory
int ProjectionToPlaneMultiMappingClass = core::RegisterObject("Compute distance between a moving point and a moving line")
        .add< ProjectionToPlaneMultiMapping< Vec3Types, Vec3Types > >()

        ;

template class SOFA_MISC_MAPPING_API ProjectionToPlaneMultiMapping< Vec3Types, Vec3Types >;





} // namespace mapping

} // namespace component

} // namespace sofa

