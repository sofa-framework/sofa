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

#include <sofa/type/config.h>

namespace sofa::type
{
template <class type, sofa::Size L>
class fixed_array;

template <sofa::Size L, class Real=float>
class Vec;

template <sofa::Size L, class Real=float>
class VecNoInit;

typedef Vec<1,float> Vec1f;
typedef Vec<1,double> Vec1d;
typedef Vec<1,int> Vec1i;
typedef Vec<1,unsigned> Vec1u;
typedef Vec<1,SReal> Vec1;

typedef Vec<2,float> Vec2f;
typedef Vec<2,double> Vec2d;
typedef Vec<2,int> Vec2i;
typedef Vec<2,unsigned> Vec2u;
typedef Vec<2,SReal> Vec2;

typedef Vec<3,float> Vec3f;
typedef Vec<3,double> Vec3d;
typedef Vec<3,int> Vec3i;
typedef Vec<3,unsigned> Vec3u;
typedef Vec<3,SReal> Vec3;

typedef Vec<4,float> Vec4f;
typedef Vec<4,double> Vec4d;
typedef Vec<4,int> Vec4i;
typedef Vec<4,unsigned> Vec4u;
typedef Vec<4,SReal> Vec4;

typedef Vec<6,float> Vec6f;
typedef Vec<6,double> Vec6d;
typedef Vec<6,int> Vec6i;
typedef Vec<6,unsigned> Vec6u;
typedef Vec<6,SReal> Vec6;

SOFA_ATTRIBUTE_DISABLED__VECTOR("Replace Vector1 with Vec1")
typedef Vec1 Vector1; ///< alias
SOFA_ATTRIBUTE_DISABLED__VECTOR("Replace Vector2 with Vec2")
typedef Vec2 Vector2; ///< alias
SOFA_ATTRIBUTE_DISABLED__VECTOR("Replace Vector3 with Vec3")
typedef Vec3 Vector3; ///< alias
SOFA_ATTRIBUTE_DISABLED__VECTOR("Replace Vector4 with Vec4")
typedef Vec4 Vector4; ///< alias
SOFA_ATTRIBUTE_DISABLED__VECTOR("Replace Vector6 with Vec6")
typedef Vec6 Vector6; ///< alias


template <sofa::Size L, sofa::Size C, class Real=float>
class Mat;

template <sofa::Size L, sofa::Size C, class Real=float>
class MatNoInit;

typedef Mat<1,1,float> Mat1x1f;
typedef Mat<1,1,double> Mat1x1d;

typedef Mat<2,2,float> Mat2x2f;
typedef Mat<2,2,double> Mat2x2d;

typedef Mat<3,3,float> Mat3x3f;
typedef Mat<3,3,double> Mat3x3d;

typedef Mat<3,4,float> Mat3x4f;
typedef Mat<3,4,double> Mat3x4d;

typedef Mat<4,4,float> Mat4x4f;
typedef Mat<4,4,double> Mat4x4d;

typedef Mat<6, 6, float> Mat6x6f;
typedef Mat<6, 6, double> Mat6x6d;

typedef Mat<2,2,SReal> Mat2x2;
typedef Mat<3,3,SReal> Mat3x3;
typedef Mat<4,4,SReal> Mat4x4;
typedef Mat<6,6,SReal> Mat6x6;

typedef Mat<2,2,SReal> Matrix2;
typedef Mat<3,3,SReal> Matrix3;
typedef Mat<4,4,SReal> Matrix4;

template <typename RealType> class Quat;
using Quatd = type::Quat<double>;
using Quatf = type::Quat<float>;

class BoundingBox;
using BoundingBox3D = BoundingBox;
class BoundingBox1D;
class BoundingBox2D;

using FixedArray1i = fixed_array<int, 1>;
using FixedArray1I = fixed_array<unsigned int, 1>;

using FixedArray2i = fixed_array<int, 2>;
using FixedArray2I = fixed_array<unsigned int, 2>;

using FixedArray3i = fixed_array<int, 3>;
using FixedArray3I = fixed_array<unsigned int, 3>;

using FixedArray4i = fixed_array<int, 4>;
using FixedArray4I = fixed_array<unsigned int, 4>;

using FixedArray5i = fixed_array<int, 5>;
using FixedArray5I = fixed_array<unsigned int, 5>;

using FixedArray6i = fixed_array<int, 6>;
using FixedArray6I = fixed_array<unsigned int, 6>;

using FixedArray7i = fixed_array<int, 7>;
using FixedArray7I = fixed_array<unsigned int, 7>;

using FixedArray8i = fixed_array<int, 8>;
using FixedArray8I = fixed_array<unsigned int, 8>;

using FixedArray1f = fixed_array<float, 1>;
using FixedArray1d = fixed_array<double, 1>;

using FixedArray2f = fixed_array<float, 2>;
using FixedArray2d = fixed_array<double, 2>;

using FixedArray3f = fixed_array<float, 3>;
using FixedArray3d = fixed_array<double, 3>;

using FixedArray4f = fixed_array<float, 4>;
using FixedArray4d = fixed_array<double, 4>;

using FixedArray5f = fixed_array<float, 5>;
using FixedArray5d = fixed_array<double, 5>;

using FixedArray6f = fixed_array<float, 6>;
using FixedArray6d = fixed_array<double, 6>;

using FixedArray7f = fixed_array<float, 7>;
using FixedArray7d = fixed_array<double, 7>;

using FixedArray8f = fixed_array<float, 8>;
using FixedArray8d = fixed_array<double, 8>;
}
