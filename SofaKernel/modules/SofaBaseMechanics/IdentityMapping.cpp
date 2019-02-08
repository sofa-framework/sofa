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
#define SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_CPP
#include <SofaBaseMechanics/IdentityMapping.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace core;
using namespace core::behavior;


// Register in the Factory
int IdentityMappingClass = core::RegisterObject("Special case of mapping where the child points are the same as the parent points")
        .add< IdentityMapping< Vec3dTypes, Vec3dTypes > >()
        .add< IdentityMapping< Vec2Types, Vec2Types > >()
        .add< IdentityMapping< Vec1Types, Vec1Types > >()
        .add< IdentityMapping< Vec6Types, Vec3dTypes > >()
        .add< IdentityMapping< Vec6Types, Vec6Types > >()
        .add< IdentityMapping< Rigid3Types, Rigid3Types > >()
        .add< IdentityMapping< Rigid2Types, Rigid2Types > >()
        .add< IdentityMapping< Vec3dTypes, ExtVec3Types > >()
        .add< IdentityMapping< Vec6Types, ExtVec3Types > >()



// Rigid -> Vec
        .add< IdentityMapping< Rigid3Types, Vec3dTypes > >()
        .add< IdentityMapping< Rigid2Types, Vec2Types > >()
        .add< IdentityMapping< Rigid3Types, ExtVec3Types > >()

        ;


template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec2Types, Vec2Types >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec1Types, Vec1Types >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6Types, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6Types, Vec6Types >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3dTypes, ExtVec3Types >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6Types, ExtVec3Types >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3Types, Rigid3Types >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2Types, Rigid2Types >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3Types, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3Types, ExtVec3Types >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2Types, Vec2Types >;





} // namespace mapping

} // namespace component

} // namespace sofa

