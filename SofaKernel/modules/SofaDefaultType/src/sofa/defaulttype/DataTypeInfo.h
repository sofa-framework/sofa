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
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/set.h>
#include <sofa/helper/types/RGBAColor.h>
#include <typeinfo>
#include "AbstractTypeInfo.h"
#include "typeinfo/DataTypeInfoDynamicWrapper.h"
#include "typeinfo/DataTypeInfo.h"
#include "typeinfo/TypeInfo_Bool.h"
#include "typeinfo/TypeInfo_Integer.h"
#include "typeinfo/TypeInfo_Mat.h"
#include "typeinfo/TypeInfo_Quat.h"
#include "typeinfo/TypeInfo_Scalar.h"
#include "typeinfo/TypeInfo_Set.h"
#include "typeinfo/TypeInfo_Text.h"
#include "typeinfo/TypeInfo_Vec.h"
#include "typeinfo/TypeInfo_FixedArray.h"
#include "typeinfo/TypeInfo_BoundingBox.h"
#include "typeinfo/TypeInfo_RGBAColor.h"
#include "typeinfo/TypeInfo_Vector.h"
#include "typeinfo/TypeInfo_RigidTypes.h"
#include "typeinfo/TypeInfo_VecTypes.h"
#include "typeinfo/TypeInfo_Topology.h"

namespace sofa::defaulttype
{

/// We make an alias to wrap around the old name to the new one.
template<class T>
using VirtualTypeInfo = DataTypeInfoDynamicWrapper<DataTypeInfo<T>>;

} /// namespace sofa::defaulttype
