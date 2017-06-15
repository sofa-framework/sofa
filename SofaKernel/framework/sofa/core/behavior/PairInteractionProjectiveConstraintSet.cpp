/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_CORE_BEHAVIOR_PAIRINTERACTIONPROJECTIVECONSTRAINTSET_CPP
#include "PairInteractionProjectiveConstraintSet.inl"

namespace sofa
{

namespace core
{

namespace behavior
{

using namespace sofa::defaulttype;
#ifndef SOFA_FLOAT
template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<Vec3dTypes>;
template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<Vec2dTypes>;
template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<Vec1dTypes>;
template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<Rigid3dTypes>;
template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<Vec3fTypes>;
template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<Vec2fTypes>;
template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<Vec1fTypes>;
template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<Rigid3fTypes>;
template class SOFA_CORE_API PairInteractionProjectiveConstraintSet<Rigid2fTypes>;
#endif

} // namespace behavior

} // namespace core

} // namespace sofa
