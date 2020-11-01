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
#include <sofa/defaulttype/typeinfo/DataTypeInfo_RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>

namespace sofa::defaulttype
{

#define REGISTER_RIGIDCOORD(theInnerType, size) template<> AbstractTypeInfo* AbstractTypeInfoCreator< sofa::defaulttype::RigidCoord<size, theInnerType> >::get() {return VirtualTypeInfoA< DataTypeInfo<sofa::defaulttype::RigidCoord<size, theInnerType>> >::get();}
#define REGISTER_RIGIDDERIV(theInnerType, size) template<> AbstractTypeInfo* AbstractTypeInfoCreator< sofa::defaulttype::RigidDeriv<size, theInnerType> >::get() {return VirtualTypeInfoA< DataTypeInfo<sofa::defaulttype::RigidDeriv<size, theInnerType>> >::get();}

REGISTER_RIGIDCOORD(double, 2)
REGISTER_RIGIDCOORD(float, 2)
REGISTER_RIGIDCOORD(double, 3)
REGISTER_RIGIDCOORD(float, 3)

REGISTER_RIGIDDERIV(double, 2)
REGISTER_RIGIDDERIV(float, 2)
REGISTER_RIGIDDERIV(double, 3)
REGISTER_RIGIDDERIV(float, 3)

} /// namespace sofa::defaulttype

