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
#define SOFA_COMPONENT_MAPPING_SUBSETMAPPING_CPP
#include "SubsetMapping.inl"

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

SOFA_DECL_CLASS(SubsetMapping)

int SubsetMappingClass = core::RegisterObject("TODO-SubsetMappingClass")
#ifndef SOFA_FLOAT
        .add< SubsetMapping< Vec3dTypes, Vec3dTypes > >()
        .add< SubsetMapping< Vec1dTypes, Vec1dTypes > >()
        .add< SubsetMapping< Vec3dTypes, ExtVec3fTypes > >()
        .add< SubsetMapping< Rigid3dTypes, Rigid3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< SubsetMapping< Vec3fTypes, Vec3fTypes > >()
        .add< SubsetMapping< Vec1fTypes, Vec1fTypes > >()
        .add< SubsetMapping< Vec3fTypes, ExtVec3fTypes > >()
        .add< SubsetMapping< Rigid3fTypes, Rigid3fTypes > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< SubsetMapping< Vec3fTypes, Vec3dTypes > >()
        .add< SubsetMapping< Vec3dTypes, Vec3fTypes > >()
        .add< SubsetMapping< Vec1fTypes, Vec1dTypes > >()
        .add< SubsetMapping< Vec1dTypes, Vec1fTypes > >()
        .add< SubsetMapping< Rigid3fTypes, Rigid3dTypes > >()
        .add< SubsetMapping< Rigid3dTypes, Rigid3fTypes > >()
#endif
#endif
        .addAlias("SurfaceIdentityMapping")
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_MECHANICS_API SubsetMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Vec1dTypes, Vec1dTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Rigid3dTypes, Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API SubsetMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Vec1fTypes, Vec1fTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Rigid3fTypes, Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API SubsetMapping< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Vec1dTypes, Vec1fTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Vec1fTypes, Vec1dTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Rigid3dTypes, Rigid3fTypes >;
template class SOFA_BASE_MECHANICS_API SubsetMapping< Rigid3fTypes, Rigid3dTypes >;
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa
