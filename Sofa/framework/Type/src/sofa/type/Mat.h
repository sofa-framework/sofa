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
#include <sofa/type/fwd.h>

#include <sofa/type/fixed_array.h>
#include <sofa/type/Vec.h>

#include <iostream>

#define EIGEN_MATRIX_PLUGIN "EigenMatrixAddons.h"
#include <Eigen/Dense>


namespace // anonymous
{
    template<typename real>
    real rabs(const real r)
    {
        if constexpr (std::is_signed<real>())
            return std::abs(r);
        else
            return r;
    }

    template<typename real>
    bool equalsZero(const real r, const real epsilon = std::numeric_limits<real>::epsilon())
    {
        return rabs(r) <= epsilon;
    }

} // anonymous namespace

namespace sofa::type
{



template <sofa::Size L, sofa::Size C, class real>
using Mat = Eigen::Matrix<real, L, C, Eigen::AutoAlign | Eigen::RowMajor>;


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

} // namespace sofa::type
