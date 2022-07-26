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
#pragma once

#include <vector>
#include <sofa/type/fixed_array.h>
#include <sofa/type/vector.h>
#include <sofa/helper/set.h>
#include <sofa/type/RGBAColor.h>
#include <typeinfo>
#include <sofa/defaulttype/AbstractTypeInfo.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfoDynamicWrapper.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Bool.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Mat.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Quat.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Set.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Text.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_BoundingBox.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_RGBAColor.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vector.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_RigidTypes.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_VecTypes.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Topology.h>

namespace sofa::defaulttype
{

/// We make an alias to wrap around the old name to the new one.
template<class T>
using VirtualTypeInfo = DataTypeInfoDynamicWrapper<DataTypeInfo<T>>;

} /// namespace sofa::defaulttype
