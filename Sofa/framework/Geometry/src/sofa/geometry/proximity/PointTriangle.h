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

#include <sofa/geometry/config.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Mat_solve_LCP.h>

#include <utility>

namespace sofa::geometry::proximity
{

template<typename Node,
     typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
     typename = std::enable_if_t<std::is_scalar_v<T>>
>
[[nodiscard]]
constexpr bool computeClosestPointOnTriangleToPoint(
    const Node& triangleP_0, const Node& triangleP_1, const Node& triangleP_2,
    const Node& pointQ,
    Node &closestPointInP)
{
    type::MatNoInit<3, 3, T> A;
    type::VecNoInit<3, T> b;
    type::VecNoInit<6, T> result;

    const Node P0P1 = triangleP_1 - triangleP_0;
    const Node P0P2 = triangleP_2 - triangleP_0;
    const Node P0Q = pointQ - triangleP_0;

    constexpr T zero = static_cast<T>(0);
    constexpr T one = static_cast<T>(1);

    A[0][2] = one;
    A[1][2] = one;
    A[2][0] = -one; A[2][1] = -one; A[2][2] = zero;
    A[0][0] = dot(P0P1,P0P1);   A[0][1] = dot(P0P2,P0P1);
    A[1][0] = dot(P0P1,P0P2);   A[1][1] = dot(P0P2,P0P2);

    b[2] = one;
    b[0] = -dot(P0Q,P0P1);
    b[1] = -dot(P0Q,P0P2);

    if (type::solveLCP(b, A, result))
    {
        const T alpha = result[3];
        const T beta = result[4];

        closestPointInP = triangleP_0 + P0P1*alpha + P0P2*beta;
        return true;
    }

    closestPointInP = triangleP_0;
    return false;
}

}
