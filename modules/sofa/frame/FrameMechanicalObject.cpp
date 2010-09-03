/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/frame/FrameMechanicalObject.h>
#include <sofa/component/container/MechanicalObject.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace container
{

using namespace sofa::defaulttype;

int MechanicalObjectClass = core::RegisterObject("mechanical state vectors")
#ifndef SOFA_FLOAT
        .add< MechanicalObject<Affine3dTypes> >()
        .add< MechanicalObject<Quadratic3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<Affine3fTypes> >()
        .add< MechanicalObject<Quadratic3fTypes> >()
#endif
        ;


template class SOFA_CORE_API MechanicalObject<Affine3dTypes>;

template class SOFA_CORE_API MechanicalObject<Affine3fTypes>;

template class SOFA_CORE_API MechanicalObject<Quadratic3dTypes>;

template class SOFA_CORE_API MechanicalObject<Quadratic3fTypes>;

} // namespace behavior

} // namespace core

} // namespace sofa
