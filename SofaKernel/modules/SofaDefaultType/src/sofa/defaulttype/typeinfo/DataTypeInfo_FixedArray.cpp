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
#include <sofa/defaulttype/typeinfo/DataTypeInfo_FixedArray.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Integer.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Bool.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Text.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_RGBAColor.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_BoundingBox.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Vec.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Vector.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>
namespace sofa::defaulttype
{

//template <typename... Ts, typename F>
//constexpr void for_types(F&& f)
//{
//    (f.template operator()<Ts>(), ...);
//}

//template<typename TT>
//int mince()
//{
//    DataTypeInfoRegistry::Set(typeid(TT), VirtualTypeInfoA< DataTypeInfo<TT>>::get());
//    return 0;
//}

//int fixedPreLoad()
//{
//    for_types<
//            char, unsigned char, int, unsigned int, long, unsigned long, long long, unsigned long long,
//            float, char,
//            std::string>([]<typename T>()
//                         {
//                             mince<sofa::helper::fixed_array<T, 1>>();
//                             mince<sofa::helper::fixed_array<T, 2>>();
//                             mince<sofa::helper::fixed_array<T, 3>>();
//                             mince<sofa::helper::fixed_array<T, 4>>();
//                             mince<sofa::helper::fixed_array<T, 5>>();
//                             mince<sofa::helper::fixed_array<T, 6>>();
//                             mince<sofa::helper::fixed_array<T, 7>>();
//                             mince<sofa::helper::fixed_array<T, 8>>();
//                             mince<sofa::helper::fixed_array<T, 9>>();
//                         });
//    return 0;
//}

//static int allFixed = fixedPreLoad();

} /// namespace sofa::defaulttype

