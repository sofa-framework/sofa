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

#include <sofa/geometry/Edge.h>

namespace sofa::geometry
{

struct Triangle
{
    static const sofa::Size NumberOfNodes = 3;

    Triangle() = delete;

    /**
    * @brief	Compute the area of a triangle
    * @remark   Depending of the type of Node, it will either use a optimized version or a generic one
    * @remark   Optimizations are enabled for sofa::type::Vec and the dimension (2D or 3D)
    * @tparam   Node iterable container (or sofa::type::Vec with cross() and norm())
    * @tparam   T scalar
    * @param	n0,n1,n2 nodes of the triangle
    * @return	Area of the triangle (a T scalar)
    */
    template<typename Node,
             typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
             typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr auto area(const Node& n0, const Node& n1, const Node& n2)
    {
        if constexpr (std::is_same_v < Node, sofa::type::Vec<3, T> >)
        {
            // half the area of a quadrilateral
            const auto a = n1 - n0;
            const auto b = n2 - n0;
            return static_cast<T>(0.5) * sofa::type::cross(a, b).norm();
        }
        else if constexpr (std::is_same_v< Node, sofa::type::Vec<2, T> >)
        {
            // shoelace formula
            return static_cast<T>(0.5) * (n0[0] * n1[1] + n1[0] * n2[1] + n2[0] * n0[1] - n1[0] * n0[1] - n2[0] * n1[1] - n0[0] * n2[1]);
        }
        else // generic without cross or diff
        {
            const auto a = sofa::geometry::Edge::length(n0, n1);
            const auto b = sofa::geometry::Edge::length(n0, n2);
            const auto c = sofa::geometry::Edge::length(n1, n2);
            return static_cast<T>(0.25) * std::sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c));
        }
    }

    /**
    * @brief	Compute the normal of a triangle
    * @remark   triangle normal computation is only possible in 3D
    * @remark   normal returned is not normalized
    * @tparam   Node iterable container (or sofa::type::Vec with cross() and norm())
    * @tparam   T scalar
    * @param	n0,n1,n2 nodes of the triangle
    * @return	Vec3 normal of this triangle
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr auto normal(const Node& n0, const Node& n1, const Node& n2)
    {
        constexpr Node n{};
        static_assert(std::distance(std::begin(n), std::end(n)) == 3, "Triangle normal can only be computed in 3 dimensions.");

        // Vec gives access to cross() and operator-
        if constexpr (std::is_same_v < Node, sofa::type::Vec<3, T> >)
        {
            const auto a = n1 - n0;
            const auto b = n2 - n0;

            return a.cross(b);
        }
        else
        {
            Node a{}, b{};
            std::transform(n1.cbegin(), n1.cend(), n0.cbegin(), a.begin(), std::minus<T>());
            std::transform(n2.cbegin(), n2.cend(), n0.cbegin(), b.begin(), std::minus<T>());

            return Node{ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2] , a[0] * b[1] - a[1] * b[0] };
        }

    }

    /**
    * @brief	Test if a ray intersects a triangle, and gives barycentric coordinates of the intersection if applicable
    * @remark   Implementation for 3D only
    * @tparam   Node iterable container (or sofa::type::Vec with cross() and norm())
    * @tparam   T scalar
    * @param	n0,n1,n2 nodes of the triangle
    * @param	t, u, v barycentric coefficients of the potential intersection in the triangle
    * @return	either if the given ray intersects the given triangle or not
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr bool rayIntersection(const Node& n0, const Node& n1, const Node& n2, const Node& origin, const Node& direction, T& t, T& u, T& v)
    {
        constexpr Node n{};
        static_assert(std::distance(std::begin(n), std::end(n)) == 3, "Ray-Triangle is only computed in 3 dimensions.");
        static_assert(std::is_same_v<Node,sofa::type::Vec<3, T> >, "rayIntersection is only implemented for sofa::type::Vec3.");

        static constexpr T epsilon = std::numeric_limits<T>::epsilon();
        static constexpr T zero = static_cast<T>(0);
        static constexpr T one = static_cast<T>(1);

        t = 0; u = 0; v = 0;

        const auto e0 = n1 - n0;
        const auto e1 = n2 - n0;

        sofa::type::Vector3 tvec, pvec, qvec;
        T det, inv_det;

        pvec = sofa::type::cross(direction, e1);

        det = sofa::type::dot(e0, pvec);
        if constexpr(std::is_floating_point_v<T>)
        {
            inv_det = one / det;
            if (std::isnan(det))
            {
                return false;
            }
        }
        else
        {
            if (std::abs(det) <= epsilon)
            {
                return false;
            }
            inv_det = one / det;
        }

        tvec = origin - n0;

        u = sofa::type::dot(tvec, pvec) * inv_det;
        if (u < zero - epsilon || u > one + epsilon)
            return false;

        qvec = sofa::type::cross(tvec, e0);

        v = sofa::type::dot(direction, qvec) * inv_det;
        if (v < zero - epsilon || (u + v) > one + epsilon)
            return false;

        t = sofa::type::dot(e1, qvec) * inv_det;

        if (t < epsilon || t != t || v != v || u != u)
            return false;

        return true;
    }

   /**
   * @brief	Test if a ray intersects a triangle
   * @remark   Implementation for 3D only
   * @tparam   Node iterable container (or sofa::type::Vec with cross() and norm())
   * @tparam   T scalar
   * @param	n0,n1,n2 nodes of the triangle
   * @return	either if the given ray intersects the given triangle or not
   */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    [[nodiscard]]
    static constexpr bool rayIntersection(const Node& n0, const Node& n1, const Node& n2, const Node& origin, const Node& direction)
    {
        constexpr Node n{};
        static_assert(std::distance(std::begin(n), std::end(n)) == 3, "Ray-Triangle is only computed in 3 dimensions.");
        static_assert(std::is_same_v<Node, sofa::type::Vec<3, T> >, "rayIntersection is only implemented for sofa::type::Vec3.");

        T t, u, v;
        return rayIntersection(n0, n1, n2, origin, direction, t, u, v);
    }

};

} // namespace sofa::geometry
