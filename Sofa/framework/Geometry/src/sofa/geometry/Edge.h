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
    * @brief	Compute the barycentric coordinates of input point on Edge (n0, n1). It can be interpreted as masses placed at the Edge vertices such that the point is the center of mass of these masses.
    * No check is done if point is on Edge. Method @sa isPointOnEdge can be used before to check that.
    * @tparam   Node iterable container
    * @tparam   T scalar
    * @param	point: position of the point to compute the coefficients
    * @param	n0,n1: nodes of the edge
    * @return	sofa::type::Vec<2, T> barycentric coefficients of each vertex of the Edge.
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
    static constexpr auto getBarycentricCoordinates(const Node& point, const Node& n0, const Node& n1)
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
        static_assert(std::is_same_v < Node, sofa::type::Vec<2, T> > || std::is_same_v < Node, sofa::type::Vec<3, T> >,
            "Method to check if point is on Edge can only be computed in 2 or 3 dimensions.");
        
        // 1. check if point is between {n0; n1}
        const auto AB = n0 - n1;
        const auto AC = n0 - p0;
        
        const auto Ln0n1 = sofa::type::dot(AB, AB);
        if (Ln0n1 < EQUALITY_THRESHOLD) // null edge
            return false;

        const auto Ln0p0 = sofa::type::dot(AB, AC);
        if (Ln0p0 < 0 || Ln0p0 > Ln0n1) // out of bounds [n0; n1]
            return false;


        // 2. check if the point, n0 and n1 are aligned
        T N2 = T(0);
        if constexpr (std::is_same_v < Node, sofa::type::Vec<2, T> >)
        {            
            N2 = sofa::type::cross(AB, AC);
        }
        else
        {
            N2 = sofa::type::cross(AB, AC).norm2();
        }

        if (N2 > EQUALITY_THRESHOLD || N2 < 0)
            return false;
        
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


    /**
    * @brief	Compute the intersection coordinate of the 2 input edges.
    * @tparam   Node iterable container
    * @tparam   T scalar
    * @param	pA, pB nodes of the first edge
    * @param	pC, pD nodes of the second edge
    * @param    intersection node will be filled if there is an intersection otherwise will return std::numeric_limits<T>::min()
    * @return	bool true if there is an intersection, otherwise false
    */
    template<typename Node,
        typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
        typename = std::enable_if_t<std::is_scalar_v<T>>
    >
        [[nodiscard]]
    static constexpr bool intersectionWithEdge(const Node& pA, const Node& pB, const Node& pC, const Node& pD, Node& intersection)
    {
        // The 2 segment equations using pX on edge1 and pY on edge2 can be defined by:
        // pX = pA + alpha (pB - pA)
        // pY = pC + beta (pD - pC)
        const auto AB = pB - pA;
        const auto CD = pD - pC;

        if constexpr (std::is_same_v < Node, sofa::type::Vec<2, T> >)
        {
            // in 2D we have 2 segment equations and 2 unknowns so direct solving of pX = pY is possible
            // pA + alpha (pB - pA) = pC + beta (pD - pC)
            // alpha = ((Cy - Ay)(Dx - Cx) - (Cx - Ax)(Dy - Cy)) / ((By - Ay)(Dx - Cx) - (Bx - Ax)(Dy - Cy))
            const auto AC = pC - pA;
            const T alphaNom = AC[1] * CD[0] - AC[0] * CD[1];
            const T alphaDenom = AB[1] * CD[0] - AB[0] * CD[1];
            
            if (alphaDenom < std::numeric_limits<T>::epsilon()) // collinear
            {
                intersection = sofa::type::Vec<2, T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::min());
                return false;
            }
            
            const T alpha = alphaNom / alphaDenom;

            if (alpha < 0 || alpha > 1)
            {
                intersection = sofa::type::Vec<2, T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::min());
                return false;
            }
            else
            {
                intersection = pA + alpha * AB;
                return true;
            }
        }
        else
        {
            // We search for the shortest line between the two 3D lines. If this lines length is null then there is an intersection
            // Shortest segment [pX; pY] between the two lines will be perpendicular to them. Then:
            // (pX - pY).dot(pB - pA) = 0
            // (pX - pY).dot(pD - pC) = 0
            
            // We need to find alpha and beta that suits: 
            // [ (pA - pC) + alpha(pB - pA) - beta(pD - pC) ].dot(pB - pA) = 0
            // [ (pA - pC) + alpha(pB - pA) - beta(pD - pC) ].dot(pD - pC) = 0
            const auto CA = pA - pC;

            // Writting d[CA/AB] == (pA - pC).dot(pB - pA) and subtituting beta we obtain:
            // beta = (d[CA/CD] + alpha * d[AB/CD]) / d[CD/CD]
            // alpha = ( d[CA/CD]*d[CD/AB] - d[CA/AB]*d[CD/CD] ) / ( d[AB/AB]*d[CD/CD] - d[AB/CD]*d[AB/CD])
            const T dCACD = sofa::type::dot(CA, CD);
            const T dABCD = sofa::type::dot(AB, CD);
            const T dCDCD = sofa::type::dot(CD, CD);
            const T dCAAB = sofa::type::dot(CA, AB);
            const T dABAB = sofa::type::dot(AB, AB);
            
            const T alphaNom = (dCACD * dABCD - dCAAB * dCDCD);
            const T alphaDenom = (dABAB * dCDCD - dABCD * dABCD); 

            if (alphaDenom < std::numeric_limits<T>::epsilon()) // alpha == inf, not sure what it means geometrically, colinear?
            {
                intersection = sofa::type::Vec<3, T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::min(), std::numeric_limits<T>::min());
                return false;
            }

            const T alpha = alphaNom / alphaDenom;
            const T beta = (dCACD + alpha * dABCD) / dCDCD;

            const Node pX = pA + alpha * AB;
            const Node pY = pC + beta * CD;

            if (alpha < 0 || beta < 0 // if alpha or beta < 0 means on the exact same line but no overlap.
                || alpha > 1 || beta > 1 // if alpha > 1 means intersection but after outside from [AB]
                || (pY - pX).norm2() > EQUALITY_THRESHOLD ) // if pY and pX are not se same means no intersection.
            {
                intersection = sofa::type::Vec<3, T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::min(), std::numeric_limits<T>::min());
                return false;
            }
            else
            {
                intersection = pX;
                return true;
            }
        }

        return false;
    }
};

} // namespace sofa::geometry
