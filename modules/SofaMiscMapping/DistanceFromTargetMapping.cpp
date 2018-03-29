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
#define SOFA_COMPONENT_MAPPING_DistanceFromTargetMapping_CPP

#include "DistanceFromTargetMapping.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(DistanceFromTargetMapping)

using namespace defaulttype;

// Register in the Factory
int DistanceFromTargetMappingClass = core::RegisterObject("Compute edge extensions")
#ifndef SOFA_FLOAT
        .add< DistanceFromTargetMapping< Vec3dTypes, Vec1dTypes > >()
        .add< DistanceFromTargetMapping< Vec1dTypes, Vec1dTypes > >()
        .add< DistanceFromTargetMapping< Rigid3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< DistanceFromTargetMapping< Vec3fTypes, Vec1fTypes > >()
        .add< DistanceFromTargetMapping< Vec1fTypes, Vec1fTypes > >()
        .add< DistanceFromTargetMapping< Rigid3fTypes, Vec1fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Vec3dTypes, Vec1dTypes >;
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Vec1dTypes, Vec1dTypes >;
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Rigid3dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Vec3fTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Vec1fTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API DistanceFromTargetMapping< Rigid3fTypes, Vec1fTypes >;
#endif




} // namespace mapping

} // namespace component

} // namespace sofa

