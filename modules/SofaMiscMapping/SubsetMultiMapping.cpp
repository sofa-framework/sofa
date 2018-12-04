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

// Register in the Factory
int SubsetMultiMappingClass = core::RegisterObject("Compute a subset of the input MechanicalObjects according to a dof index list")
    .add< SubsetMultiMapping< Vec3Types, Vec3Types > >()
    .add< SubsetMultiMapping< Vec1Types, Vec1Types > >()
    .add< SubsetMultiMapping< Rigid3Types, Rigid3Types > >()
    .add< SubsetMultiMapping< Rigid3Types, Vec3Types > >()

        ;

template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Vec3Types, Vec3Types >;
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Vec1Types, Vec1Types >;
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Rigid3Types, Rigid3Types >;
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Rigid3Types, Vec3Types >;


} // namespace mapping

} // namespace component

} // namespace sofa
