/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/core/componentmodel/behavior/MixedInteractionForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

using namespace sofa::defaulttype;


template class MixedInteractionForceField<Vec3dTypes, Vec3dTypes>;
template class MixedInteractionForceField<Vec2dTypes, Vec2dTypes>;
template class MixedInteractionForceField<Vec1dTypes, Vec1dTypes>;
template class MixedInteractionForceField<Rigid3dTypes, Rigid3dTypes> ;
template class MixedInteractionForceField<Rigid2dTypes, Rigid2dTypes> ;
template class MixedInteractionForceField<Vec3dTypes, Rigid3dTypes> ;
template class MixedInteractionForceField<Vec2dTypes, Rigid2dTypes> ;
template class MixedInteractionForceField<Rigid3dTypes, Vec3dTypes> ;
template class MixedInteractionForceField<Rigid2dTypes, Vec2dTypes> ;

template class MixedInteractionForceField<Vec3fTypes, Vec3fTypes>;
template class MixedInteractionForceField<Vec2fTypes, Vec2fTypes>;
template class MixedInteractionForceField<Vec1fTypes, Vec1fTypes>;
template class MixedInteractionForceField<Rigid3fTypes, Rigid3fTypes> ;
template class MixedInteractionForceField<Rigid2fTypes, Rigid2fTypes> ;
template class MixedInteractionForceField<Vec3fTypes, Rigid3fTypes> ;
template class MixedInteractionForceField<Vec2fTypes, Rigid2fTypes> ;
template class MixedInteractionForceField<Rigid3fTypes, Vec3fTypes> ;
template class MixedInteractionForceField<Rigid2fTypes, Vec2fTypes> ;

template class MixedInteractionForceField<Vec3dTypes, Vec3fTypes>;
template class MixedInteractionForceField<Vec2dTypes, Vec2fTypes>;
template class MixedInteractionForceField<Vec1dTypes, Vec1fTypes>;
template class MixedInteractionForceField<Rigid3dTypes, Rigid3fTypes> ;
template class MixedInteractionForceField<Rigid2dTypes, Rigid2fTypes> ;
template class MixedInteractionForceField<Vec3dTypes, Rigid3fTypes> ;
template class MixedInteractionForceField<Vec2dTypes, Rigid2fTypes> ;
template class MixedInteractionForceField<Rigid3dTypes, Vec3fTypes> ;
template class MixedInteractionForceField<Rigid2dTypes, Vec2fTypes> ;


template class MixedInteractionForceField<Vec3fTypes, Vec3dTypes>;
template class MixedInteractionForceField<Vec2fTypes, Vec2dTypes>;
template class MixedInteractionForceField<Vec1fTypes, Vec1dTypes>;
template class MixedInteractionForceField<Rigid3fTypes, Rigid3dTypes> ;
template class MixedInteractionForceField<Rigid2fTypes, Rigid2dTypes> ;
template class MixedInteractionForceField<Vec3fTypes, Rigid3dTypes> ;
template class MixedInteractionForceField<Vec2fTypes, Rigid2dTypes> ;
template class MixedInteractionForceField<Rigid3fTypes, Vec3dTypes> ;
template class MixedInteractionForceField<Rigid2fTypes, Vec2dTypes> ;


} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa
