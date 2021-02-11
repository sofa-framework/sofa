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
template <sofa::Size L, sofa::Size C, class Real>
class Mat;

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

typedef Mat<2,2,SReal> Mat2x2;
typedef Mat<3,3,SReal> Mat3x3;
typedef Mat<4,4,SReal> Mat4x4;

typedef Mat<2,2,SReal> Matrix2;
typedef Mat<3,3,SReal> Matrix3;
typedef Mat<4,4,SReal> Matrix4;
}

/// Deprecated namespace.
namespace sofa::defaulttype
{
using sofa::type::Mat;
using sofa::type::Mat1x1f;
using sofa::type::Mat1x1d;

using sofa::type::Mat2x2f;
using sofa::type::Mat2x2d;

using sofa::type::Mat3x3f;
using sofa::type::Mat3x3d;

using sofa::type::Mat3x4f;
using sofa::type::Mat3x4d;

using sofa::type::Mat4x4f;
using sofa::type::Mat4x4d;

using sofa::type::Mat2x2;
using sofa::type::Mat3x3;
using sofa::type::Mat4x4;

using sofa::type::Matrix2;
using sofa::type::Matrix3;
using sofa::type::Matrix4;
}
