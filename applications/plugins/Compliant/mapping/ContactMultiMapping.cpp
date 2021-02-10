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
#define SOFA_COMPONENT_MAPPING_CONTACTMULTIMAPPING_CPP

#include "ContactMultiMapping.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(ContactMultiMapping)

using namespace defaulttype;

// Register in the Factory
int ContactMultiMappingClass = core::RegisterObject("Maps relative position/velocity between contact points")
    .add< ContactMultiMapping< Vec3Types, Vec1Types > >()
    .add< ContactMultiMapping< Vec3Types, Vec2Types > >()
    .add< ContactMultiMapping< Vec3Types, Vec3Types > >()
;

template class SOFA_Compliant_API ContactMultiMapping<  Vec3Types, Vec1Types >;
template class SOFA_Compliant_API ContactMultiMapping<  Vec3Types, Vec2Types >;
template class SOFA_Compliant_API ContactMultiMapping<  Vec3Types, Vec3Types >;

} // namespace mapping

} // namespace component

} // namespace sofa


