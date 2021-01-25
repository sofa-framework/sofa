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

#include <sofa/type/MatSym.h>

SOFA_DEPRECATED_HEADER(v21.12, "sofa/type/MatSym.h")

namespace sofa::defaulttype
{
    template <std::size_t N, typename real = float>
    using MatSym = sofa::type::MatSym<N, real>;

    //template<class real>
    //inline real determinant(const Mat<2, 2, real>& m)
    //{
    //    return type::determinant(m);
    //}

    //template<class real>
    //inline real determinant(const Mat<2, 3, real>& m)
    //{
    //    return type::determinant(m);
    //}

    //template<class real>
    //inline real determinant(const Mat<3, 2, real>& m)
    //{
    //    return type::determinant(m);
    //}

    //template<class real>
    //inline real oneNorm(const Mat<3, 3, real>& A)
    //{
    //    return type::oneNorm(A);
    //}

    //template<class real>
    //inline real infNorm(const Mat<3, 3, real>& A)
    //{
    //    return type::infNorm(A);
    //}

    //template<sofa::Size N, class real>
    //inline real trace(const Mat<N, N, real>& m)
    //{
    //    return type::trace(m);
    //}

    //template<sofa::Size N, class real>
    //inline Vec<N, real> diagonal(const Mat<N, N, real>& m)
    //{
    //    return type::diagonal(m);
    //}

    ///// Matrix inversion (general case).
    //template<sofa::Size S, class real>
    //bool invertMatrix(Mat<S, S, real>& dest, const Mat<S, S, real>& from)
    //{
    //    return type::invertMatrix(dest, from);
    //}

    //template<class real>
    //bool invertMatrix(Mat<3, 3, real>& dest, const Mat<3, 3, real>& from)
    //{
    //    return type::invertMatrix(dest, from);
    //}

    //template<class real>
    //bool invertMatrix(Mat<2, 2, real>& dest, const Mat<2, 2, real>& from)
    //{
    //    return type::invertMatrix(dest, from);
    //}

    //template<sofa::Size S, class real>
    //bool transformInvertMatrix(Mat<S, S, real>& dest, const Mat<S, S, real>& from)
    //{
    //    return type::transformInvertMatrix(dest, from);
    //}

    using Mat1x1f = sofa::type::Mat1x1f;
    using Mat1x1d = sofa::type::Mat1x1d;

    using Mat2x2f = sofa::type::Mat2x2f;
    using Mat2x2d = sofa::type::Mat2x2d;

    using Mat3x3f = sofa::type::Mat3x3f;
    using Mat3x3d = sofa::type::Mat3x3d;

    using Mat3x4f = sofa::type::Mat3x4f;
    using Mat3x4d = sofa::type::Mat3x4d;

    using Mat4x4f = sofa::type::Mat4x4f;
    using Mat4x4d = sofa::type::Mat4x4d;

    using Mat2x2 = sofa::type::Mat2x2;
    using Mat3x3 = sofa::type::Mat3x3;
    using Mat4x4 = sofa::type::Mat4x4;

    using Matrix2 = sofa::type::Matrix2;
    using Matrix3 = sofa::type::Matrix3;
    using Matrix4 = sofa::type::Matrix4;

} // namespace sofa::defaulttype
