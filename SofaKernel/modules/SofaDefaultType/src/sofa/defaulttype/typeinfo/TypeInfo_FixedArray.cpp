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
#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Bool.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Text.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_RGBAColor.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_BoundingBox.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vector.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>
namespace sofa::defaulttype
{

template <typename... Ts, typename F>
constexpr void for_types(F&& f)
{
    (f.template operator()<Ts>(), ...);
}

template<typename TT>
int loadIt()
{
    DataTypeInfoRegistry::Set(DataTypeId<TT>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<TT>>::get(), sofa_tostring(SOFA_TARGET));
    return 0;
}

int fixedPreLoad()
{
    for_types<
            char, unsigned char, int, unsigned int, long, unsigned long, long long, unsigned long long,
            float, double, char,
            std::string>([]<typename T>()
                         {
                             loadIt<sofa::helper::fixed_array<T, 1>>();
                             loadIt<sofa::helper::fixed_array<T, 2>>();
                             loadIt<sofa::helper::fixed_array<T, 3>>();
                             loadIt<sofa::helper::fixed_array<T, 4>>();
                             loadIt<sofa::helper::fixed_array<T, 5>>();
                             loadIt<sofa::helper::fixed_array<T, 6>>();
                             loadIt<sofa::helper::fixed_array<T, 7>>();
                             loadIt<sofa::helper::fixed_array<T, 8>>();
                             loadIt<sofa::helper::fixed_array<T, 9>>();
                         });
    return 0;
}

static int allFixed = fixedPreLoad();

} /// namespace sofa::defaulttype

