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
#define SOFA_CORE_BEHAVIOR_CONSTRAINT_CPP
#include <sofa/core/behavior/Constraint.inl>

namespace sofa::core::behavior
{

using namespace sofa::defaulttype;
template class SOFA_CORE_API Constraint<Vec6Types>;
template class SOFA_CORE_API Constraint<Vec3Types>;
template class SOFA_CORE_API Constraint<Vec2Types>;
template class SOFA_CORE_API Constraint<Vec1Types>;
template class SOFA_CORE_API Constraint<Rigid3Types>;
template class SOFA_CORE_API Constraint<Rigid2Types>;


} // namespace sofa::core::behavior
