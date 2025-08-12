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
#include <sofa/geometry/ElementType.h>
#include <sofa/type/fixed_array.h>
#include <sofa/type/Mat.h>
#include <cmath>
#include <numeric>
#include <iterator>
#include <array>

#include <sofa/geometry/Tetrahedron.h>

namespace sofa::geometry
{

struct Hexahedron
{
    static constexpr sofa::Size NumberOfNodes = 8;
    static constexpr ElementType Element_type = ElementType::HEXAHEDRON;

    Hexahedron() = delete;

    // CONVENTION : indices ordering for the nodes of an hexahedron :
    //
    //     Y  n3---------n2
    //     ^  /          /|
    //     | /          / |
    //     n7---------n6  |
    //     |          |   |
    //     |  n0------|--n1
    //     | /        | /
    //     |/         |/
    //     n4---------n5-->X
    //    /
    //   /
    //  Z

    /**
    * @brief	Compute the center of a hexahedron
    * @remark	The order of nodes given as parameter is not necessary.
    * @tparam   Node iterable container, with operator[]
    * @tparam   T scalar
    * @param	n0,n1,n2,n3,n4,n5,n6,n7 nodes of the hexahedron
    * @return	Center of the hexahedron (same type as the given nodes)
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr auto center(const Node& n0, const Node& n1, const Node& n2, const Node& n3, 
                                 const Node& n4, const Node& n5, const Node& n6, const Node& n7)
    {
        constexpr auto dimensions = sizeof(Node) / sizeof(T);
        auto centerRes = n0;
        for (size_t i = 0; i < dimensions; i++)
        {
            centerRes[i] += n1[i] + n2[i] + n3[i] + n4[i] + n5[i] + n6[i] + n7[i];
            centerRes[i] /= static_cast<T>(NumberOfNodes);
        }

        return centerRes;
    }

    /**
    * @brief	Compute the barycentric coefficients of a node in a hexahedron
    * @remark	Due to some optimizations, the order of nodes given as parameter is necessary.
    * @tparam   Node iterable container, with operator[]
    * @tparam   T scalar
    * @param	n0,n1,n2,n3,n4,n5,n6,n7 nodes of the hexahedron
    * @param    pos position of the node from which the coefficients will be computed 
    * @return	A Vec3 container with the barycentric coefficients of the given node
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr auto barycentricCoefficients(const Node& n0, const Node& n1, const Node& n2, const Node& n3,
                                                  const Node& n4, const Node& n5, const Node& n6, const Node& n7, 
                                                  const Node& pos)
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

        return tmpResult;
    }

    /**
    * @brief	Compute the squared distance between a node and the center of a hexahedron
    * @remark	Due to some optimizations, the order of nodes given as parameter is necessary.
    * @tparam   Node iterable container, with operator[]
    * @tparam   T scalar
    * @param	n0,n1,n2,n3,n4,n5,n6,n7 nodes of the hexahedron
    * @param    pos position of the node from which the distance will be computed
    * @return	Distance from the node and the center of the hexahedron, as a T scalar
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr auto squaredDistanceTo(const Node& n0, const Node& n1, const Node& n2, const Node& n3,
                                            const Node& n4, const Node& n5, const Node& n6, const Node& n7, 
                                            const Node& pos)
    {
        const auto& v = barycentricCoefficients(n0,n1,n2,n3,n4,n5,n6,n7, pos);

        T d = std::max(std::max(-v[0], -v[1]), std::max(std::max(-v[2], v[0] - 1), std::max(v[1] - static_cast<T>(1), v[2] - static_cast<T>(1))));

        if (d > 0)
            d = (pos - center(n0, n1, n2, n3, n4, n5, n6, n7)).norm2();
        else
            d = static_cast<T>(0);

        return d;
    }

    /**
    * @brief	Compute a position from a given set of barycentric coefficients and the associated hexahedron
    * @remark	The order of nodes given as parameter is necessary.
    * @tparam   Node iterable container, with operator* applicable with a scalar
    * @tparam   T scalar
    * @param	n0,n1,n2,n3,n4,n5,n6,n7 nodes of the hexahedron
    * @param    baryC barycentric coefficients
    * @return	Position computed from the coefficients, as a Node type
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr auto getPositionFromBarycentricCoefficients(const Node& n0, const Node& n1, const Node& n2, const Node& n3,
            const Node& n4, const Node& n5, const Node& n6, const Node& n7, const sofa::type::fixed_array<SReal, 3>& baryC)
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

    /**
    * @brief	Compute the volume of a hexahedron
    * @remark	non optimized version: just return the sum of the 6 inner-tetrahedra
    * @tparam   Node iterable container
    * @tparam   T scalar
    * @param	n0,n1,n2,n3,n4,n5,n6,n7 nodes of the hexahedron
    * @return	Volume of the hexahedron, as a T scalar
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr auto volume(const Node& n0, const Node& n1, const Node& n2, const Node& n3,
                                 const Node& n4, const Node& n5, const Node& n6, const Node& n7)
    {
        constexpr Node n{};
        static_assert(std::distance(std::begin(n), std::end(n)) == 3, "volume can only be computed in 3 dimensions.");

        return sofa::geometry::Tetrahedron::volume(n0, n5, n1, n6)
             + sofa::geometry::Tetrahedron::volume(n0, n1, n3, n6)
             + sofa::geometry::Tetrahedron::volume(n1, n3, n6, n2)
             + sofa::geometry::Tetrahedron::volume(n6, n3, n0, n7)
             + sofa::geometry::Tetrahedron::volume(n6, n7, n0, n5)
             + sofa::geometry::Tetrahedron::volume(n7, n5, n4, n0);

    }
};

} // namespace sofa::geometry
