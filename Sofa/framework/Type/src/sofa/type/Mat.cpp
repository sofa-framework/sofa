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
#include <sofa/type/Mat.h>

namespace sofa::type
{

template<typename Real>
constexpr Mat<3,3, Real> mult33(const Mat<3,3,Real>& m1, const Mat<3,3,Real>& m2)
{
    Mat<3,3,Real> r(NOINIT);

    const auto A00 = m1[0][0];
    const auto A01 = m1[0][1];
    const auto A02 = m1[0][2];
    const auto A10 = m1[1][0];
    const auto A11 = m1[1][1];
    const auto A12 = m1[1][2];
    const auto A20 = m1[2][0];
    const auto A21 = m1[2][1];
    const auto A22 = m1[2][2];

    const auto B00 = m2[0][0];
    const auto B01 = m2[0][1];
    const auto B02 = m2[0][2];
    const auto B10 = m2[1][0];
    const auto B11 = m2[1][1];
    const auto B12 = m2[1][2];
    const auto B20 = m2[2][0];
    const auto B21 = m2[2][1];
    const auto B22 = m2[2][2];

    r[0][0] = A00 * B00 + A01 * B10 + A02 * B20;
    r[0][1] = A00 * B01 + A01 * B11 + A02 * B21;
    r[0][2] = A00 * B02 + A01 * B12 + A02 * B22;

    r[1][0] = A10 * B00 + A11 * B10 + A12 * B20;
    r[1][1] = A10 * B01 + A11 * B11 + A12 * B21;
    r[1][2] = A10 * B02 + A11 * B12 + A12 * B22;

    r[2][0] = A20 * B00 + A21 * B10 + A22 * B20;
    r[2][1] = A20 * B01 + A21 * B11 + A22 * B21;
    r[2][2] = A20 * B02 + A21 * B12 + A22 * B22;

    return r;
}

template<>
template<> SOFA_TYPE_API
constexpr Mat<3,3,float> Mat<3,3,float>::operator*(const Mat<3,3,float>& m) const noexcept
{
    return mult33(*this, m);
}

template<>
template<> SOFA_TYPE_API
constexpr Mat<3,3,double> Mat<3,3,double>::operator*(const Mat<3,3,double>& m) const noexcept
{
    return mult33(*this, m);
}

template class SOFA_TYPE_API Mat<2, 2, SReal>;
template class SOFA_TYPE_API Mat<2, 3, SReal>;
template class SOFA_TYPE_API Mat<3, 3, SReal>;
template class SOFA_TYPE_API Mat<4, 4, SReal>;
template class SOFA_TYPE_API Mat<6, 3, SReal>;
template class SOFA_TYPE_API Mat<6, 6, SReal>;
template class SOFA_TYPE_API Mat<8, 3, SReal>;
template class SOFA_TYPE_API Mat<8, 8, SReal>;
template class SOFA_TYPE_API Mat<9, 9, SReal>;
template class SOFA_TYPE_API Mat<12, 3, SReal>;
template class SOFA_TYPE_API Mat<12, 6, SReal>;
template class SOFA_TYPE_API Mat<12, 12, SReal>;
template class SOFA_TYPE_API Mat<20, 20, SReal>;
template class SOFA_TYPE_API Mat<20, 32, SReal>;
template class SOFA_TYPE_API Mat<24, 24, SReal>;
template class SOFA_TYPE_API Mat<32, 20, SReal>;
}
