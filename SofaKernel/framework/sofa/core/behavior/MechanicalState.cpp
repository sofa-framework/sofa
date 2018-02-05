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
#define SOFA_CORE_BEHAVIOR_MECHANICALSTATE_CPP
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa
{

namespace core
{

namespace behavior
{

using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
template class SOFA_CORE_API MechanicalState<Vec3dTypes>;
template class SOFA_CORE_API MechanicalState<Vec2dTypes>;
template class SOFA_CORE_API MechanicalState<Vec1dTypes>;
template class SOFA_CORE_API MechanicalState<Vec6dTypes>;
template class SOFA_CORE_API MechanicalState<Rigid3dTypes>;
template class SOFA_CORE_API MechanicalState<Rigid2dTypes>;
template class SOFA_CORE_API MechanicalState<ExtVec3dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_CORE_API MechanicalState<Vec3fTypes>;
template class SOFA_CORE_API MechanicalState<Vec2fTypes>;
template class SOFA_CORE_API MechanicalState<Vec1fTypes>;
template class SOFA_CORE_API MechanicalState<Vec6fTypes>;
template class SOFA_CORE_API MechanicalState<Rigid3fTypes>;
template class SOFA_CORE_API MechanicalState<Rigid2fTypes>;
template class SOFA_CORE_API MechanicalState<ExtVec3fTypes>;
#endif

} // namespace behavior

} // namespace core

} // namespace sofa
