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
#include <sofa/type/Vec.h>
#include <cmath>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <type_traits>

namespace sofa::geometry
{

struct Edge
{
    static constexpr sofa::Size NumberOfNodes = 2;

    Edge() = delete;

    /**
    * @brief	Compute the squared length (or norm) of an edge
    * @remark   Depending of the type of Node, it will either use a optimized version or a generic one
    * @remark   Optimizations are enabled for sofa::type::Vec
    * @tparam   Node iterable container (or sofa::type::Vec for operator- and norm2())
    * @tparam   T scalar
    * @param	n0,n1 nodes of the edge
    * @return	Squared length of the edge (a T scalar)
    */
    template<typename Node,
             typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
             typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr auto squaredLength(const Node& n0, const Node& n1)
    {
        constexpr Node v{};
        constexpr auto size = std::distance(std::cbegin(v), std::cend(v));

        // specialized function is faster than the generic (using STL) one
        if constexpr (std::is_same_v< Node, sofa::type::Vec<size, T>>)
        {
            return (static_cast<sofa::type::Vec<size, T>>(n1) - static_cast<sofa::type::Vec<size, T>>(n0)).norm2();
        }
        else
        {
            Node diff{};
            std::transform(n0.cbegin(), n0.cend(), n1.cbegin(), diff.begin(), std::minus<T>());
            return std::inner_product(std::cbegin(diff), std::cend(diff), std::cbegin(diff), static_cast<T>(0));
        }
    }

    /**
    * @brief	Compute the length (or norm) of an edge
    * @remark   Depending of the type of Node, it will either use a optimized version or a generic one
    * @remark   Optimizations are enabled for sofa::type::Vec
    * @tparam   Node iterable container (or sofa::type::Vec for squaredLength())
    * @tparam   T scalar
    * @param	n0,n1 nodes of the edge
    * @return	Length of the edge (a T scalar)
    */
    template<typename Node,
             typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
             typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    static constexpr auto length(const Node& n0, const Node& n1)
    {
        return std::sqrt(squaredLength(n0, n1));
    }


    /**
    * @brief	Compute the barycentric coefficients of input point on Edge (n0, n1)
    * @tparam   Node iterable container
    * @tparam   T scalar
    * @param	point: position of the point to compute the coefficients
    * @param	n0,n1: nodes of the edge
    * @return	sofa::type::Vec<2, T> barycentric coefficients
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    static constexpr auto pointBaryCoefs(const Node& point, const Node& n0, const Node& n1)
    {
        sofa::type::Vec<2, T> baryCoefs;
        const T dist = (n1 - n0).norm();

        if (dist < EQUALITY_THRESHOLD)
        {
            baryCoefs[0] = 0.5;
            baryCoefs[1] = 0.5;
        }
        else
        {
            baryCoefs[0] = (point - n1).norm() / dist;
            baryCoefs[1] = (point - n0).norm() / dist;
        }

        return baryCoefs;
    }


    /**
    * @brief	Test if a point is on Edge (n0, n1)
    * @tparam   Node iterable container
    * @tparam   T scalar
    * @param	p0: position of the point to test
    * @param	n0,n1: nodes of the edge
    * @return	bool result if point is on Edge.
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
        static constexpr bool isPointOnEdge(const Node& p0, const Node& n0, const Node& n1)
    {
        // 1. check if the point, n0 and n1 are aligned
        const auto AB = n0 - n1;
        const auto AC = n0 - p0;

        T N2 = T(0);
        if constexpr (std::is_same_v < Node, sofa::type::Vec<2, T> >)
        {            
            N2 = sofa::type::cross(AB, AC);
        }
        else
        {
            N2 = sofa::type::cross(AB, AC).norm2();
        }

        if (N2 > EQUALITY_THRESHOLD || N2 < 0.0f)
            return false;


        // 2. check if point is between {n0; n1}
        const auto Ln0p0 = sofa::type::dot(AB, AC);
        const auto Ln0n1 = sofa::type::dot(AB, AB);

        if (Ln0p0 < 0.0f || Ln0p0 > Ln0n1) // out of bounds [n0; n1]
            return false;
        else if (Ln0n1 < EQUALITY_THRESHOLD) // null edge
            return false;
        else
            return true;
    }


    /**
    * @brief	Compute the intersection between a plane (defined by a point and a normal) and the Edge (n0, n1)
    * @tparam   Node iterable container
    * @tparam   T scalar
    * @param	n0,n1 nodes of the edge
    * @param	planeP0,normal position and normal defining the plan
    * @param    intersection position of the intersection (if one) between the plane and the Edge
    * @return	bool true if there is an intersection, otherwise false
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr bool intersectionWithPlane(const Node& n0, const Node& n1, const sofa::type::Vec<3, T>& planeP0, const sofa::type::Vec<3, T>& normal, sofa::type::Vec<3, T>& intersection)
    {
        constexpr Node n{}; 
        static_assert(std::distance(std::begin(n), std::end(n)) == 3, "Plane - Edge intersection can only be computed in 3 dimensions.");

        //plane equation
        const sofa::type::Vec<3, T> planeNorm = normal.normalized();
        const T d = planeNorm * planeP0;

        //compute intersection between line and plane equation
        const T denominator = planeNorm * (n1 - n0);
        if (denominator < EQUALITY_THRESHOLD)
        {
            return false;
        }

        const T t = (d - planeNorm * n0) / denominator;
        if ((t <= 1) && (t >= 0))
        {
            intersection = n0 + (n1 - n0) * t;
            return true;
        }
        return false;
    }
};

} // namespace sofa::geometry
