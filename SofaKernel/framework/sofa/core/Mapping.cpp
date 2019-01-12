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
#define SOFA_CORE_MAPPING_CPP
#include "Mapping.inl"
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace core
{

using namespace sofa::defaulttype;
using namespace core;

template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3dTypes >;

template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec1Types >;

template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec2Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec1Types >;

template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6Types, sofa::defaulttype::Vec6Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6Types, sofa::defaulttype::Vec3dTypes >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6Types, sofa::defaulttype::Vec1Types >;

// Rigid templates
template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Rigid2Types >;

template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec6Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec3dTypes >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec1Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Rigid3Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Rigid3Types >;

// This one is special: ExtVec3Types are used for output outside of Sofa.
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2Types, sofa::defaulttype::ExtVec2Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6Types, sofa::defaulttype::ExtVec3Types >;
template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::ExtVec3Types >;


// cross templates


} // namespace core

} // namespace sofa

