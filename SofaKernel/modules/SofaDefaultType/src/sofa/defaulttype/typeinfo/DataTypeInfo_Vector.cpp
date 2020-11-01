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
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Bool.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Text.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_RGBAColor.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_FixedArray.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_BoundingBox.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Vec.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Vector.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>

namespace sofa::defaulttype
{

template <typename... Ts, typename F>
constexpr void for_types(F&& f)
{
    (f.template operator()<Ts>(), ...);
}

int vectorPreLoad()
{
    for_types<char, unsigned char, int, unsigned int, long, unsigned long, long long, unsigned long long,
            float, double,
            std::string,
            Vec1f, Vec1d, Vec1i, Vec1u,
            Vec2f, Vec2d, Vec2i, Vec2u,
            Vec3f, Vec3d, Vec3i, Vec3u,
            Vec4f, Vec4d, Vec4i, Vec4u,
            sofa::helper::types::RGBAColor, BoundingBox>([]<typename T>()
                                                         {
//                                                             DataTypeInfoRegistry::Set(DataTypeId<T>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<T> >::get());
                                                             DataTypeInfoRegistry::Set(DataTypeId<std::vector<T>>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<std::vector<T>> >::get());
                                                             DataTypeInfoRegistry::Set(DataTypeId<sofa::helper::vector<T>>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::helper::vector<T>> >::get());

                                                             typedef sofa::helper::fixed_array<float, 1> r;
                                                             DataTypeInfoRegistry::Set(DataTypeId<std::vector<r>>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<std::vector<r>> >::get());
                                                             DataTypeInfoRegistry::Set(DataTypeId<sofa::helper::vector<r>>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::helper::vector<r>> >::get());

                                                             typedef sofa::helper::fixed_array<float, 2> rr;
                                                             DataTypeInfoRegistry::Set(DataTypeId<std::vector<rr>>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<std::vector<rr>> >::get());
                                                             DataTypeInfoRegistry::Set(DataTypeId<sofa::helper::vector<rr>>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::helper::vector<rr>> >::get());


                                                             typedef sofa::helper::fixed_array<int, 1> r1;
                                                             DataTypeInfoRegistry::Set(DataTypeId<std::vector<r1>>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<std::vector<r>> >::get());
                                                             DataTypeInfoRegistry::Set(DataTypeId<sofa::helper::vector<r1>>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::helper::vector<r>> >::get());

                                                             typedef sofa::helper::fixed_array<int, 2> r2;
                                                             DataTypeInfoRegistry::Set(DataTypeId<std::vector<r2>>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<std::vector<rr>> >::get());
                                                             DataTypeInfoRegistry::Set(DataTypeId<sofa::helper::vector<r2>>::getTypeId(), VirtualTypeInfoA< DataTypeInfo<sofa::helper::vector<rr>> >::get());
                                                         });
    return 0;
}

static int allVector = vectorPreLoad();
} /// namespace sofa::defaulttype

