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
#define SOFA_CORE_BEHAVIOR_MIXEDINTERACTIONFORCEFIELD_CPP
#include <sofa/core/behavior/MixedInteractionForceField.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace core
{

namespace behavior
{

using namespace sofa::defaulttype;
#ifndef SOFA_FLOAT
template class SOFA_CORE_API MixedInteractionForceField<Vec1dTypes, Vec3dTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec1dTypes, Rigid3dTypes>;

template class SOFA_CORE_API MixedInteractionForceField<Vec3dTypes, Vec3dTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec2dTypes, Vec2dTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec1dTypes, Vec1dTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Rigid3dTypes, Rigid3dTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid2dTypes, Rigid2dTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Vec3dTypes, Rigid3dTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Vec2dTypes, Rigid2dTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid3dTypes, Vec3dTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid2dTypes, Vec2dTypes> ;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CORE_API MixedInteractionForceField<Vec1fTypes, Vec3fTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec1fTypes, Rigid3fTypes>;

template class SOFA_CORE_API MixedInteractionForceField<Vec3fTypes, Vec3fTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec2fTypes, Vec2fTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec1fTypes, Vec1fTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Rigid3fTypes, Rigid3fTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid2fTypes, Rigid2fTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Vec3fTypes, Rigid3fTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Vec2fTypes, Rigid2fTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid3fTypes, Vec3fTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid2fTypes, Vec2fTypes> ;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CORE_API MixedInteractionForceField<Vec1dTypes, Vec3fTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec1fTypes, Vec3dTypes>;

template class SOFA_CORE_API MixedInteractionForceField<Vec1dTypes, Rigid3fTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec1fTypes, Rigid3dTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec3dTypes, Vec3fTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec2dTypes, Vec2fTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec1dTypes, Vec1fTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Rigid3dTypes, Rigid3fTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid2dTypes, Rigid2fTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Vec3dTypes, Rigid3fTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Vec2dTypes, Rigid2fTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid3dTypes, Vec3fTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid2dTypes, Vec2fTypes> ;

template class SOFA_CORE_API MixedInteractionForceField<Vec3fTypes, Vec3dTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec2fTypes, Vec2dTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Vec1fTypes, Vec1dTypes>;
template class SOFA_CORE_API MixedInteractionForceField<Rigid3fTypes, Rigid3dTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid2fTypes, Rigid2dTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Vec3fTypes, Rigid3dTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Vec2fTypes, Rigid2dTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid3fTypes, Vec3dTypes> ;
template class SOFA_CORE_API MixedInteractionForceField<Rigid2fTypes, Vec2dTypes> ;
#endif
#endif

} // namespace behavior

} // namespace core

} // namespace sofa
