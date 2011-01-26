/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "PendulumMapping.inl"
#include <sofa/core/Mapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;



SOFA_DECL_CLASS(PendulumMapping)


int PendulumMappingClass = core::RegisterObject("just an example of templated component")
#ifndef SOFA_FLOAT
        .add< PendulumMapping<Vec1dTypes,Vec3dTypes> >()
        .add< PendulumMapping<Vec1dTypes,Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< PendulumMapping<Vec1fTypes,Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class PendulumMapping<Vec1dTypes,Vec3dTypes>;
template class PendulumMapping<Vec1dTypes,Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class PendulumMapping<Vec1fTypes,Vec3fTypes>;
#endif



}	//mapping

}	//component

}	//sofa

