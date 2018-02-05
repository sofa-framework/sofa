/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_COMPONENT_MAPPING_DistanceMapping_CPP

#include "DistanceMapping.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(DistanceMapping)
SOFA_DECL_CLASS(DistanceMultiMapping)


using namespace defaulttype;


// Register in the Factory
int DistanceMappingClass = core::RegisterObject("Compute edge extensions")
#ifndef SOFA_FLOAT
        .add< DistanceMapping< Vec3dTypes, Vec1dTypes > >()
        .add< DistanceMapping< Rigid3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< DistanceMapping< Vec3fTypes, Vec1fTypes > >()
        .add< DistanceMapping< Rigid3fTypes, Vec1fTypes > >()
#endif
        ;
int DistanceMultiMappingClass = core::RegisterObject("Compute edge extensions")
#ifndef SOFA_FLOAT
        .add< DistanceMultiMapping< Vec3dTypes, Vec1dTypes > >()
        .add< DistanceMultiMapping< Rigid3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< DistanceMultiMapping< Vec3fTypes, Vec1fTypes > >()
        .add< DistanceMultiMapping< Rigid3fTypes, Vec1fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_MAPPING_API DistanceMapping< Vec3dTypes, Vec1dTypes >;
template class SOFA_MISC_MAPPING_API DistanceMapping< Rigid3dTypes, Vec1dTypes >;
template class SOFA_MISC_MAPPING_API DistanceMultiMapping< Vec3dTypes, Vec1dTypes >;
template class SOFA_MISC_MAPPING_API DistanceMultiMapping< Rigid3dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API DistanceMapping< Vec3fTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API DistanceMapping< Rigid3fTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API DistanceMultiMapping< Vec3fTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API DistanceMultiMapping< Rigid3fTypes, Vec1fTypes >;
#endif




} // namespace mapping

} // namespace component

} // namespace sofa

