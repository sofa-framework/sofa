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

#include <type_traits>

namespace sofa::geometry::proximity
{

template<typename Node,
     typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
     typename = std::enable_if_t<std::is_scalar_v<T>>
>
[[nodiscard]]
constexpr bool computeClosestPointsInTwoTriangles(
    const Node& triangleP_0, const Node& triangleP_1, const Node& triangleP_2,
    const Node& triangleQ_0, const Node& triangleQ_1, const Node& triangleQ_2,
    Node &closestPointInP, Node &closestPointInQ)
{
    type::MatNoInit<6, 6, T> A;
    type::VecNoInit<6, T> b;
    type::VecNoInit<12, T> result;

    const Node P0P1 { triangleP_1 - triangleP_0 };
    const Node P0P2 { triangleP_2 - triangleP_0 };
    const Node Q0Q1 { triangleQ_1 - triangleQ_0 };
    const Node Q0Q2 { triangleQ_2 - triangleQ_0 };
    const Node P0Q0 { triangleQ_0 - triangleP_0 };

    constexpr T zero = static_cast<T>(0);
    constexpr T one = static_cast<T>(1);

    A[0][4] = one; A[4][0] = -one;
    A[1][4] = one; A[4][1] = -one;
    A[2][4] = zero; A[4][2] = zero;
    A[3][4] = zero; A[4][3] = zero;

    A[0][5] = zero; A[5][0] = zero;
    A[1][5] = zero; A[5][1] = zero;
    A[2][5] = one; A[5][2] = -one;
    A[3][5] = one; A[5][3] = -one;

    A[4][4] = zero; A[5][5] = zero;
    A[4][5] = zero; A[5][4] = zero;

    A[0][0] =  dot(P0P1,P0P1);  A[0][1] =  dot(P0P2,P0P1);  A[0][2] = -dot(Q0Q1,P0P1);  A[0][3] = -dot(Q0Q2,P0P1);
    A[1][0] =  dot(P0P1,P0P2);  A[1][1] =  dot(P0P2,P0P2);  A[1][2] = -dot(Q0Q1,P0P2);  A[1][3] = -dot(Q0Q2,P0P2);
    A[2][0] = -dot(P0P1,Q0Q1);  A[2][1] = -dot(P0P2,Q0Q1);  A[2][2] =  dot(Q0Q1,Q0Q1);  A[2][3] =  dot(Q0Q2,Q0Q1);
    A[3][0] = -dot(P0P1,Q0Q2);  A[3][1] = -dot(P0P2,Q0Q2);  A[3][2] =  dot(Q0Q1,Q0Q2);  A[3][3] =  dot(Q0Q2,Q0Q2);

    b[0]=-dot(P0Q0,P0P1);
    b[1]=-dot(P0Q0,P0P2);
    b[2]= dot(P0Q0,Q0Q1);
    b[3]= dot(P0Q0,Q0Q2);
    b[4]= one;
    b[5]= one;

    if (type::solveLCP(b, A, result))
    {
        const T alphaP = result[6];
        const T betaP = result[7];
        const T alphaQ = result[8];
        const T betaQ = result[9];
        closestPointInP = triangleP_0 + P0P1 * alphaP + P0P2 * betaP;
        closestPointInQ = triangleQ_0 + Q0Q1 * alphaQ + Q0Q2 * betaQ;
        return true;
    }

    closestPointInP = triangleP_0;
    closestPointInQ = triangleQ_0;
    return false;
}

}
