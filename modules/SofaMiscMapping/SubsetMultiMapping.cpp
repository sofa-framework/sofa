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
#define SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_CPP

#include <SofaMiscMapping/SubsetMultiMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

using namespace sofa::defaulttype;

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(SubsetMultiMapping)

// Register in the Factory
int SubsetMultiMappingClass = core::RegisterObject("Compute a subset of the input MechanicalObjects according to a dof index list")
#ifndef SOFA_FLOAT
    .add< SubsetMultiMapping< Vec3dTypes, Vec3dTypes > >()
    .add< SubsetMultiMapping< Vec1dTypes, Vec1dTypes > >()
    .add< SubsetMultiMapping< Rigid3dTypes, Rigid3dTypes > >()
    .add< SubsetMultiMapping< Rigid3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
    .add< SubsetMultiMapping< Vec3fTypes, Vec3fTypes > >()
    .add< SubsetMultiMapping< Vec1fTypes, Vec1fTypes > >()
    .add< SubsetMultiMapping< Rigid3fTypes, Rigid3fTypes > >()
    .add< SubsetMultiMapping< Rigid3fTypes, Vec3fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Vec1dTypes, Vec1dTypes >;
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Rigid3dTypes, Rigid3dTypes >;
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Rigid3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Vec1fTypes, Vec1fTypes >;
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Rigid3fTypes, Rigid3fTypes >;
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Rigid3fTypes, Vec3fTypes >;
#endif

} // namespace mapping

} // namespace component

} // namespace sofa
