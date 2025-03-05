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
#define SOFA_CORE_ACCUMULATIONVECID_CPP
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/core/AccumulationVecId.inl>

namespace sofa::core
{
template struct SOFA_CORE_API AccumulationVecId<defaulttype::Vec3dTypes, V_DERIV, V_READ>;
template struct SOFA_CORE_API AccumulationVecId<defaulttype::Vec2Types, V_DERIV, V_READ>;
template struct SOFA_CORE_API AccumulationVecId<defaulttype::Vec1Types, V_DERIV, V_READ>;
template struct SOFA_CORE_API AccumulationVecId<defaulttype::Vec6Types, V_DERIV, V_READ>;
template struct SOFA_CORE_API AccumulationVecId<defaulttype::Rigid3Types, V_DERIV, V_READ>;
template struct SOFA_CORE_API AccumulationVecId<defaulttype::Rigid2Types, V_DERIV, V_READ>;
template struct SOFA_CORE_API AccumulationVecId<defaulttype::Vec3fTypes, V_DERIV, V_READ>;
}
