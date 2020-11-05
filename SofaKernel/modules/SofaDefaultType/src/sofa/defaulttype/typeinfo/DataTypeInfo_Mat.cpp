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
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Mat.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Vec.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Scalar.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>

namespace sofa::defaulttype
{

static int Mat1x1fDataTypeInfo = DataTypeInfoRegistry::Set(DataTypeId<sofa::defaulttype::Mat1x1f>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::defaulttype::Mat1x1f> >::get(), sofa_tostring(SOFA_TARGET));
static int Mat1x1dDataTypeInfo = DataTypeInfoRegistry::Set(DataTypeId<sofa::defaulttype::Mat1x1d>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::defaulttype::Mat1x1d> >::get(), sofa_tostring(SOFA_TARGET));
static int Mat2x2fDataTypeInfo = DataTypeInfoRegistry::Set(DataTypeId<sofa::defaulttype::Mat2x2f>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::defaulttype::Mat2x2f> >::get(), sofa_tostring(SOFA_TARGET));
static int Mat2x2dDataTypeInfo = DataTypeInfoRegistry::Set(DataTypeId<sofa::defaulttype::Mat2x2d>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::defaulttype::Mat2x2d> >::get(), sofa_tostring(SOFA_TARGET));
static int Mat3x3fDataTypeInfo = DataTypeInfoRegistry::Set(DataTypeId<sofa::defaulttype::Mat3x3f>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::defaulttype::Mat3x3f> >::get(), sofa_tostring(SOFA_TARGET));
static int Mat3x3dDataTypeInfo = DataTypeInfoRegistry::Set(DataTypeId<sofa::defaulttype::Mat3x3d>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::defaulttype::Mat3x3d> >::get(), sofa_tostring(SOFA_TARGET));
static int Mat4x4fDataTypeInfo = DataTypeInfoRegistry::Set(DataTypeId<sofa::defaulttype::Mat4x4f>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::defaulttype::Mat4x4f> >::get(), sofa_tostring(SOFA_TARGET));
static int Mat4x4dDataTypeInfo = DataTypeInfoRegistry::Set(DataTypeId<sofa::defaulttype::Mat4x4d>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::defaulttype::Mat4x4d> >::get(), sofa_tostring(SOFA_TARGET));

} /// namespace sofa::defaulttype

