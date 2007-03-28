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
#include <sofa/component/forcefield/PlaneForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/MechanicalObject.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class PlaneForceField<Vec3dTypes>;
template class PlaneForceField<Vec3fTypes>;
template class PlaneForceField<Vec2dTypes>;
template class PlaneForceField<Vec2fTypes>;
template class PlaneForceField<Vec1dTypes>;
template class PlaneForceField<Vec1fTypes>;
template class PlaneForceField<Vec6dTypes>;
template class PlaneForceField<Vec6fTypes>;


SOFA_DECL_CLASS(PlaneForceField)

int PlaneForceFieldClass = core::RegisterObject("Repulsion applied by a plane toward the exterior (half-space)")
        .add< PlaneForceField<Vec3dTypes> >()
        .add< PlaneForceField<Vec3fTypes> >()
        .add< PlaneForceField<Vec2dTypes> >()
        .add< PlaneForceField<Vec2fTypes> >()
        .add< PlaneForceField<Vec1dTypes> >()
        .add< PlaneForceField<Vec1fTypes> >()
        .add< PlaneForceField<Vec6dTypes> >()
        .add< PlaneForceField<Vec6fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa
