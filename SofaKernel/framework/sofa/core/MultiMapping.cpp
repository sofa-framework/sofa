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
#define SOFA_CORE_MULTIMAPPING_CPP
#include <sofa/core/MultiMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace core
{

using namespace sofa::defaulttype;
using namespace core::behavior;

#ifndef SOFA_FLOAT
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec1dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec2dTypes, sofa::defaulttype::Vec1dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec2dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec1dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec6dTypes, sofa::defaulttype::Vec1dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec1dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec3dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec6dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Rigid3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec1fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec2fTypes, sofa::defaulttype::Vec1fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec2fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec1fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec6fTypes, sofa::defaulttype::Vec1fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec1fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec3fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec6fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec1fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec1dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec2dTypes, sofa::defaulttype::Vec1fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec2fTypes, sofa::defaulttype::Vec1dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec2fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec1fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec2dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec1dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec1fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec3fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec6fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Rigid3fTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec1dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec3dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec6dTypes >;
template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Rigid3dTypes >;
#endif
#endif

} // namespace core

} // namespace sofa
