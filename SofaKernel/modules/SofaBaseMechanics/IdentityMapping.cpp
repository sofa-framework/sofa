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


SOFA_DECL_CLASS(IdentityMapping)

// Register in the Factory
int IdentityMappingClass = core::RegisterObject("Special case of mapping where the child points are the same as the parent points")
#ifndef SOFA_FLOAT
        .add< IdentityMapping< Vec3dTypes, Vec3dTypes > >()
        .add< IdentityMapping< Vec2dTypes, Vec2dTypes > >()
        .add< IdentityMapping< Vec1dTypes, Vec1dTypes > >()
        .add< IdentityMapping< Vec6dTypes, Vec3dTypes > >()
        .add< IdentityMapping< Vec6dTypes, Vec6dTypes > >()
        .add< IdentityMapping< Rigid3dTypes, Rigid3dTypes > >()
        .add< IdentityMapping< Rigid2dTypes, Rigid2dTypes > >()
        .add< IdentityMapping< Vec3dTypes, ExtVec3fTypes > >()
        .add< IdentityMapping< Vec6dTypes, ExtVec3fTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< IdentityMapping< Vec3fTypes, Vec3fTypes > >()
        .add< IdentityMapping< Vec2fTypes, Vec2fTypes > >()
        .add< IdentityMapping< Vec1fTypes, Vec1fTypes > >()
        .add< IdentityMapping< Vec6fTypes, Vec6fTypes > >()
        .add< IdentityMapping< Rigid3fTypes, Rigid3fTypes > >()
        .add< IdentityMapping< Rigid2fTypes, Rigid2fTypes > >()
        .add< IdentityMapping< Vec3fTypes, ExtVec3fTypes > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< IdentityMapping< Vec3fTypes, Vec3dTypes > >()
        .add< IdentityMapping< Vec3dTypes, Vec3fTypes > >()
        .add< IdentityMapping< Vec2fTypes, Vec2dTypes > >()
        .add< IdentityMapping< Vec2dTypes, Vec2fTypes > >()
        .add< IdentityMapping< Vec1fTypes, Vec1dTypes > >()
        .add< IdentityMapping< Vec1dTypes, Vec1fTypes > >()
        .add< IdentityMapping< Vec6fTypes, Vec6dTypes > >()
        .add< IdentityMapping< Vec6dTypes, Vec6fTypes > >()
        .add< IdentityMapping< Rigid3dTypes, Rigid3fTypes > >()
        .add< IdentityMapping< Rigid3fTypes, Rigid3dTypes > >()
        .add< IdentityMapping< Rigid2dTypes, Rigid2fTypes > >()
        .add< IdentityMapping< Rigid2fTypes, Rigid2dTypes > >()
#endif
#endif

// Rigid -> Vec
#ifndef SOFA_FLOAT
        .add< IdentityMapping< Rigid3dTypes, Vec3dTypes > >()
        .add< IdentityMapping< Rigid2dTypes, Vec2dTypes > >()
        .add< IdentityMapping< Rigid3dTypes, ExtVec3fTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< IdentityMapping< Rigid3fTypes, Vec3fTypes > >()
        .add< IdentityMapping< Rigid2fTypes, Vec2fTypes > >()
        .add< IdentityMapping< Rigid3fTypes, ExtVec3fTypes > >()
#endif
        ;


#ifndef SOFA_FLOAT
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec2dTypes, Vec2dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec1dTypes, Vec1dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6dTypes, Vec6dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3dTypes, Rigid3dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2dTypes, Rigid2dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2dTypes, Vec2dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec2fTypes, Vec2fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec1fTypes, Vec1fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6fTypes, Vec6fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3fTypes, Rigid3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2fTypes, Rigid2fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2fTypes, Vec2fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec2dTypes, Vec2fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec2fTypes, Vec2dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec1dTypes, Vec1fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec1fTypes, Vec1dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6dTypes, Vec6fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6fTypes, Vec6dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Vec6fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3fTypes, Rigid3dTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid3dTypes, Rigid3fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2dTypes, Rigid2fTypes >;
template class SOFA_BASE_MECHANICS_API IdentityMapping< Rigid2fTypes, Rigid2dTypes >;
#endif
#endif


} // namespace mapping

} // namespace component

} // namespace sofa

