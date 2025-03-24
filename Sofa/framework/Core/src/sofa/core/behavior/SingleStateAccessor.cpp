﻿/******************************************************************************
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
#define SOFA_CORE_BEHAVIOR_SINGLESTATEACCESSOR_CPP

#include <sofa/core/behavior/SingleStateAccessor.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::core::behavior
{

using namespace sofa::defaulttype;

template class SOFA_CORE_API SingleStateAccessor<Vec1Types>;
template class SOFA_CORE_API SingleStateAccessor<Vec2Types>;
template class SOFA_CORE_API SingleStateAccessor<Vec3Types>;
template class SOFA_CORE_API SingleStateAccessor<Vec6Types>;
template class SOFA_CORE_API SingleStateAccessor<Rigid2Types>;
template class SOFA_CORE_API SingleStateAccessor<Rigid3Types>;

}
