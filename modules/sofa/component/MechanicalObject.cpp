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
#include <sofa/component/MechanicalObject.inl>

#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/mass/UniformMass.h>

#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa
{

namespace component
{

using namespace core::componentmodel::behavior;
using namespace defaulttype;

SOFA_DECL_CLASS(MechanicalObject)

int MechanicalObjectClass = core::RegisterObject("mechanical state vectors")
        .add< MechanicalObject<Vec3dTypes> >(true) // default template
        .add< MechanicalObject<Vec3fTypes> >()
        .add< MechanicalObject<Rigid3dTypes> >()
        .add< MechanicalObject<Rigid3fTypes> >()
        .add< MechanicalObject<LaparoscopicRigid3Types> >()
        .add< MechanicalObject<Vec2dTypes> >()
        .add< MechanicalObject<Vec2fTypes> >()
        .add< MechanicalObject<Rigid2dTypes> >()
        .add< MechanicalObject<Rigid2fTypes> >()
        .add< MechanicalObject<Vec1dTypes> >()
        .add< MechanicalObject<Vec1fTypes> >()
        .add< MechanicalObject<Vec6dTypes> >()
        .add< MechanicalObject<Vec6fTypes> >()
        ;

// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.

template class MechanicalObject<defaulttype::Vec3fTypes>;
template class MechanicalObject<defaulttype::Vec3dTypes>;
template class MechanicalObject<defaulttype::Vec2fTypes>;
template class MechanicalObject<defaulttype::Vec2dTypes>;
template class MechanicalObject<defaulttype::Vec1fTypes>;
template class MechanicalObject<defaulttype::Vec1dTypes>;

template class MechanicalObject<defaulttype::Rigid3dTypes>;
template class MechanicalObject<defaulttype::Rigid3fTypes>;
template class MechanicalObject<defaulttype::Rigid2dTypes>;
template class MechanicalObject<defaulttype::Rigid2fTypes>;

template class MechanicalObject<defaulttype::LaparoscopicRigid3Types>;

template<>
void MechanicalObject<defaulttype::Rigid3dTypes>::applyRotation (const defaulttype::Quat q)
{
    VecCoord& x = *this->getX();
    for (unsigned int i = 0; i < x.size(); i++)
        x[i].getOrientation() *= q;
}

template<>
void MechanicalObject<defaulttype::Rigid3fTypes>::applyRotation (const defaulttype::Quat q)
{
    VecCoord& x = *this->getX();
    for (unsigned int i = 0; i < x.size(); i++)
        x[i].getOrientation() *= q;
}

} // namespace component

} // namespace sofa
