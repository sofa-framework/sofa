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
    static constexpr sofa::Size NumberOfNodes = 3;

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
    * @brief	Compute the barycentric coordinates of the input point in the Triangle. It can be interpreted as masses placed at the vertices of Triangle (n0, n1, n2), such that the point is the center of mass of these masses.
    * @tparam   Node iterable container
    * @tparam   T scalar
    * @param	p0: position of the input point to compute the coefficients
    * @param	n0, n1, n2: nodes of the triangle
    * @return	sofa::type::Vec<3, T> barycentric coefficients of each vertex of the Triangle. These masses can be zero or negative; they are all positive if and only if the point is inside the Triangle. 
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
        static constexpr auto getBarycentricCoordinates(const Node& p0, const Node& n0, const Node& n1, const Node& n2)
    {
        // Point can be written: p0 = a*n0 + b*n1 + c*n2
        // with a = area(n1n2p0)/area(n0n1n2), b = area(n0n2p0)/area(n0n1n2) and c = area(n0n1p0)/area(n0n1n2) 
        const auto area = Triangle::area(n0, n1, n2);
        if (fabs(area) < std::numeric_limits<T>::epsilon()) // triangle is flat
        {
            return sofa::type::Vec<3, T>(-1, -1, -1);
        }
        
        const auto A0 = Triangle::area(n1, n2, p0);
        const auto A1 = Triangle::area(n0, p0, n2);

        sofa::type::Vec<3, T> baryCoefs(type::NOINIT);
        baryCoefs[0] = A0 / area;
        baryCoefs[1] = A1 / area;
        baryCoefs[2] = 1 - baryCoefs[0] - baryCoefs[1];

        if (fabs(baryCoefs[2]) <= std::numeric_limits<T>::epsilon()){
            baryCoefs[2] = 0;
        }
        
        return baryCoefs;
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
    * @brief	Test if input point is inside Triangle (n0, n1, n2) using Triangle @sa getBarycentricCoordinates . The point is inside the Triangle if and only if Those coordinates are all positive.
    * @tparam   Node iterable container
    * @tparam   T scalar
    * @param	p0: position of the point to test
    * @param	n0, n1, n2: nodes of the triangle
    * @param	output parameter: sofa::type::Vec<3, T> barycentric coordinates of the input point in Triangle
    * @return	bool result if point is inside Triangle.
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
        static constexpr bool isPointInTriangle(const Node& p0, const Node& n0, const Node& n1, const Node& n2, sofa::type::Vec<3, T>& baryCoefs)
    {
        baryCoefs = Triangle::getBarycentricCoordinates(p0, n0, n1, n2);

        for (int i = 0; i < 3; ++i)
        {
            if (baryCoefs[i] < 0 || baryCoefs[i] > 1)
                return false;
        }

        return true;
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
    template<typename TReal>
    [[nodiscard]]
    static constexpr bool rayIntersection(const sofa::type::Vec<3, TReal>& n0, const sofa::type::Vec<3, TReal>& n1, const sofa::type::Vec<3, TReal>& n2, const sofa::type::Vec<3, TReal>& origin, const sofa::type::Vec<3, TReal>& direction, TReal& t, TReal& u, TReal& v)
    {

        constexpr TReal epsilon = std::numeric_limits<TReal>::epsilon();
        constexpr TReal zero = static_cast<TReal>(0);
        constexpr TReal one = static_cast<TReal>(1);

        t = 0; u = 0; v = 0;

        const auto e0 = n1 - n0;
        const auto e1 = n2 - n0;

        sofa::type::Vec<3, TReal> tvec(type::NOINIT);
        sofa::type::Vec<3, TReal> pvec(type::NOINIT);
        sofa::type::Vec<3, TReal> qvec(type::NOINIT);
        TReal det;
        TReal inv_det;

        pvec = sofa::type::cross(direction, e1);

        det = sofa::type::dot(e0, pvec);

        if (std::fabs(det) <= epsilon)
        {
            return false;
        }

        inv_det = one / det;

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
    template<typename TReal>
    [[nodiscard]]
    static constexpr bool rayIntersection(const sofa::type::Vec<3, TReal>& n0, const sofa::type::Vec<3, TReal>& n1, const sofa::type::Vec<3, TReal>& n2, const sofa::type::Vec<3, TReal>& origin, const sofa::type::Vec<3, TReal>& direction)
    {
        TReal t, u, v;
        return rayIntersection(n0, n1, n2, origin, direction, t, u, v);
    }

};

} // namespace sofa::geometry
