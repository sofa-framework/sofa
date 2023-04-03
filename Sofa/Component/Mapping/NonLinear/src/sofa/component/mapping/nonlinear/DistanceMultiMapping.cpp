/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#define SOFA_COMPONENT_MAPPING_DistanceMultiMapping_CPP

#include <sofa/core/ObjectFactory.h>
#include <sofa/component/mapping/nonlinear/DistanceMultiMapping.inl>

namespace sofa::component::mapping::nonlinear
{

using defaulttype::Vec3Types;
using defaulttype::Vec1Types;
using defaulttype::Rigid3Types;

int DistanceMultiMappingClass = core::RegisterObject("Compute edge extensions")
        .add< DistanceMultiMapping< Vec3Types, Vec1Types > >()
        .add< DistanceMultiMapping< Rigid3Types, Vec1Types > >()
        ;

template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceMultiMapping< Vec3Types, Vec1Types >;
template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceMultiMapping< Rigid3Types, Vec1Types >;
}
