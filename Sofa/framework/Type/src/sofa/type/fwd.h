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
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>

namespace sofa::type
{
template <class type, sofa::Size L>
class fixed_array;

template <sofa::Size L, sofa::Size C, class Real=float>
class MatNoInit;

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
