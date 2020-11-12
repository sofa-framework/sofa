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
#include <sofa/defaulttype/typeinfo/TypeInfo_Bool.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Text.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_RGBAColor.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_BoundingBox.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vector.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>

namespace sofa::defaulttype
{
using sofa::helper::fixed_array;
using sofa::helper::vector;

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
            fixed_array<float, 1>, fixed_array<float, 2>, fixed_array<float, 3>, fixed_array<float, 4>, fixed_array<float, 5>, fixed_array<float, 6>, fixed_array<float, 7>, fixed_array<float, 8>,
            fixed_array<double, 1>, fixed_array<double, 2>, fixed_array<double, 3>, fixed_array<double, 4>, fixed_array<double, 5>, fixed_array<double, 6>, fixed_array<double, 7>, fixed_array<double, 8>,
            fixed_array<char, 1>, fixed_array<char, 2>, fixed_array<char, 3>, fixed_array<char, 4>, fixed_array<char, 5>, fixed_array<char, 6>, fixed_array<char, 7>, fixed_array<double, 8>,
            fixed_array<unsigned char, 1>, fixed_array<unsigned char, 2>, fixed_array<unsigned char, 3>, fixed_array<unsigned char, 4>, fixed_array<unsigned char, 5>, fixed_array<unsigned char, 6>, fixed_array<unsigned char, 7>, fixed_array<unsigned char, 8>,
            fixed_array<int, 1>, fixed_array<int, 2>, fixed_array<int, 3>, fixed_array<int, 4>, fixed_array<int, 5>, fixed_array<int, 6>, fixed_array<int, 7>, fixed_array<int, 8>,
            fixed_array<unsigned int, 1>, fixed_array<unsigned int, 2>, fixed_array<unsigned int, 3>, fixed_array<unsigned int, 4>, fixed_array<unsigned int, 5>, fixed_array<unsigned int, 6>, fixed_array<unsigned int, 7>, fixed_array<unsigned int, 8>,
            fixed_array<long, 1>, fixed_array<long, 2>, fixed_array<long, 3>, fixed_array<long, 4>, fixed_array<long, 5>, fixed_array<long, 6>, fixed_array<long, 7>, fixed_array<long, 8>,
            fixed_array<unsigned long, 1>, fixed_array<unsigned int, 2>, fixed_array<unsigned long, 3>, fixed_array<unsigned long, 4>, fixed_array<unsigned long, 5>, fixed_array<unsigned long, 6>, fixed_array<unsigned long, 7>, fixed_array<unsigned long, 8>,
            Rigid2dMass, Rigid2dTypes, Rigid2fMass, Rigid2fTypes,
            Rigid3dMass, Rigid3dTypes, Rigid3fMass, Rigid3fTypes,
            Rigid3dTypes::Coord, Rigid3dTypes::Deriv, Rigid3fTypes::Coord, Rigid3fTypes::Deriv,
            Rigid2dTypes::Coord, Rigid2dTypes::Deriv, Rigid2fTypes::Coord, Rigid2fTypes::Deriv,
            sofa::helper::types::RGBAColor, BoundingBox>([]<typename T>()
                                                         {
                                                             DataTypeInfoRegistry::Set(DataTypeId<vector<T>>::getTypeId(),
                                                                                       VirtualTypeInfoA< DataTypeInfo<vector<T>> >::get(), sofa_tostring(SOFA_TARGET) );
                                                             DataTypeInfoRegistry::Set(DataTypeId<vector<vector<T>>>::getTypeId(),
                                                                                       VirtualTypeInfoA< DataTypeInfo<vector<vector<T>>> >::get(), sofa_tostring(SOFA_TARGET));
                                                               });
    return 0;
}

static int allVector = vectorPreLoad();
} /// namespace sofa::defaulttype

