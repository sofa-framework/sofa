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
#define SOFA_COMPONENT_MAPPING_SquareDistanceMapping_CPP

#include "SquareDistanceMapping.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(SquareDistanceMapping)


using namespace defaulttype;


// Register in the Factory
int SquareDistanceMappingClass = core::RegisterObject("Compute square edge extensions")
#ifndef SOFA_FLOAT
        .add< SquareDistanceMapping< Vec3dTypes, Vec1dTypes > >()
        .add< SquareDistanceMapping< Rigid3dTypes, Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< SquareDistanceMapping< Vec3fTypes, Vec1fTypes > >()
        .add< SquareDistanceMapping< Rigid3fTypes, Vec1fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_MAPPING_API SquareDistanceMapping< Vec3dTypes, Vec1dTypes >;
template class SOFA_MISC_MAPPING_API SquareDistanceMapping< Rigid3dTypes, Vec1dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API SquareDistanceMapping< Vec3fTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API SquareDistanceMapping< Rigid3fTypes, Vec1fTypes >;
#endif




} // namespace mapping

} // namespace component

} // namespace sofa

