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
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/MappedObject.inl>

namespace sofa
{

namespace component
{

using namespace core::componentmodel::behavior;
using namespace defaulttype;

SOFA_DECL_CLASS(MappedObject)

int MappedObjectClass = core::RegisterObject("Mapped state vectors")
        .add< MappedObject<Vec3dTypes> >(true) // default template
        .add< MappedObject<Vec3fTypes> >()
        .add< MappedObject<Rigid3dTypes> >()
        .add< MappedObject<Rigid3fTypes> >()
        .add< MappedObject<LaparoscopicRigid3Types> >()
        .add< MappedObject<Vec2dTypes> >()
        .add< MappedObject<Vec2fTypes> >()
        .add< MappedObject<Rigid2dTypes> >()
        .add< MappedObject<Rigid2fTypes> >()
        .add< MappedObject<Vec1dTypes> >()
        .add< MappedObject<Vec1fTypes> >()
        .add< MappedObject<Vec6dTypes> >()
        .add< MappedObject<Vec6fTypes> >()
        ;

// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.

template class MappedObject<defaulttype::Vec3fTypes>;
template class MappedObject<defaulttype::Vec3dTypes>;
template class MappedObject<defaulttype::Vec2fTypes>;
template class MappedObject<defaulttype::Vec2dTypes>;
template class MappedObject<defaulttype::Vec1fTypes>;
template class MappedObject<defaulttype::Vec1dTypes>;

template class MappedObject<defaulttype::Rigid3dTypes>;
template class MappedObject<defaulttype::Rigid3fTypes>;
template class MappedObject<defaulttype::Rigid2dTypes>;
template class MappedObject<defaulttype::Rigid2fTypes>;

template class MappedObject<defaulttype::LaparoscopicRigid3Types>;

} // namespace component

} // namespace sofa
