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
#include <sofa/type/fixed_array.h>
#include <sofa/type/Mat.h>
#include <cmath>
#include <numeric>
#include <iterator>
#include <array>

namespace sofa::geometry
{

struct Hexahedron
{
    static constexpr sofa::Size NumberOfNodes = 8;

    Hexahedron() = default;

    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
        static constexpr auto center(const Node& n0, const Node& n1, const Node& n2, const Node& n3, 
                                     const Node& n4, const Node& n5, const Node& n6, const Node& n7)
    {
        constexpr auto dimensions = sizeof(Node) / sizeof(T);
        auto centerRes = n0;
        for (auto i = 0; i < dimensions; i++)
        {
            centerRes[i] += n1[i] + n2[i] + n3[i] + n4[i] + n5[i] + n6[i] + n7[i];
            centerRes[i] /= static_cast<T>(NumberOfNodes);
        }

        return centerRes;
    }

    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
        static constexpr auto barycentricCoefficients(const Node& n0, const Node& n1, const Node& n2, const Node& n3,
            const Node& n4, const Node& n5, const Node& n6, const Node& n7, const Node& pos)
    {
        SOFA_UNUSED(n2);
        SOFA_UNUSED(n5);
        SOFA_UNUSED(n6);
        SOFA_UNUSED(n7);
        constexpr sofa::Size dimensions = sizeof(Node) / sizeof(T);
        constexpr auto max_spatial_dimensions = std::min(3u, dimensions);

        sofa::type::Vec<3, T> origin, p1, p3, p4, pnt;

        for (unsigned int w = 0; w < max_spatial_dimensions; ++w)
        {
            origin[w] = n0[w];
            p1[w] = n1[w];
            p3[w] = n3[w];
            p4[w] = n4[w];
            pnt[w] = pos[w];
        }

        sofa::type::Mat<3,3,T> m, mt, base;
        m[0] = p1 - origin;
        m[1] = p3 - origin;
        m[2] = p4 - origin;
        mt.transpose(m);
        const bool canInvert = base.invert(mt);
        assert(canInvert);
        SOFA_UNUSED(canInvert);
        const auto tmpResult = base * (pnt - origin);

        if constexpr (std::is_same_v<decltype(tmpResult), Node>)
        {
            return tmpResult;
        }
        else
        {
            sofa::type::fixed_array<T, 3> returnResult{};
            std::copy_n(tmpResult.begin(), max_spatial_dimensions, returnResult.begin());
            return returnResult;
        }
    }

    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
        static constexpr auto distanceTo(const Node& n0, const Node& n1, const Node& n2, const Node& n3,
            const Node& n4, const Node& n5, const Node& n6, const Node& n7, const Node& pos)
    {
        const auto& v = barycentricCoefficients(n0,n1,n2,n3,n4,n5,n6,n7, pos);

        T d = std::max(std::max(-v[0], -v[1]), std::max(std::max(-v[2], v[0] - 1), std::max(v[1] - static_cast<T>(1), v[2] - static_cast<T>(1))));

        if (d > 0)
            d = (pos - center(n0, n1, n2, n3, n4, n5, n6, n7)).norm2();

        return d;
    }

    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    static constexpr auto getPositionFromBarycentricCoefficients(const Node& n0, const Node& n1, const Node& n2, const Node& n3,
            const Node& n4, const Node& n5, const Node& n6, const Node& n7, const std::array<SReal, 3>& baryC)
    {
        const auto fx = baryC[0];
        const auto fy = baryC[1];
        const auto fz = baryC[2];

        const auto pos = n0 * ((1 - fx) * (1 - fy) * (1 - fz))
            + n1 * ((fx) * (1 - fy) * (1 - fz))
            + n3 * ((1 - fx) * (fy) * (1 - fz))
            + n2 * ((fx) * (fy) * (1 - fz))
            + n4 * ((1 - fx) * (1 - fy) * (fz))
            + n5 * ((fx) * (1 - fy) * (fz))
            + n7 * ((1 - fx) * (fy) * (fz))
            + n6 * ((fx) * (fy) * (fz));

        return pos;
    }

    

};

} // namespace sofa::geometry
