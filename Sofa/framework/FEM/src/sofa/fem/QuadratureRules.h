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
#include <sofa/fem/config.h>

#include <sofa/geometry/Edge.h>
#include <sofa/geometry/Triangle.h>
#include <sofa/geometry/Quad.h>
#include <sofa/geometry/Tetrahedron.h>
#include <sofa/geometry/Hexahedron.h>
#include <sofa/geometry/Prism.h>
#include <sofa/geometry/Pyramid.h>
#include <sofa/geometry/QuadraticElements.h>
#include <sofa/type/Vec.h>

#include <array>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace sofa::fem
{

/// Mapping element types to their integration reference domain. 
/// This domain is the same irrespective of the order of the element.
template <class ElementType>
struct ReferenceDomain
{
    using type = ElementType;
};

template <> struct ReferenceDomain<sofa::geometry::QuadraticEdge>        { using type = sofa::geometry::Edge; };
template <> struct ReferenceDomain<sofa::geometry::QuadraticTriangle>    { using type = sofa::geometry::Triangle; };
template <> struct ReferenceDomain<sofa::geometry::QuadraticQuad>        { using type = sofa::geometry::Quad; };
template <> struct ReferenceDomain<sofa::geometry::QuadraticTetrahedron> { using type = sofa::geometry::Tetrahedron; };
template <> struct ReferenceDomain<sofa::geometry::QuadraticHexahedron>  { using type = sofa::geometry::Hexahedron; };

template <class ElementType>
using ReferenceDomain_t = typename ReferenceDomain<ElementType>::type;

/// Always false, but dependent on its argument: will trigger compilation failure in case no rule
/// has been tabulated for the requested degree.
template <sofa::Size>
inline constexpr bool degreeIsNotTabulated = false;

/**
 * Quadrature rules over the reference domain of a topological element.
 *
 * An incomplete type, followed by specializations. Each specialization exposes:
 *  - TopologicalDimension, the dimension of the reference domain;
 *  - tabulatedDegrees, the polynomial degrees for which a table is provided, ascending;
 *  - points<Degree, Real>(), the table for one of those degrees.
 *
 * A request for a degree that is not tabulated is served by the smallest tabulated
 * degree above it; see QuadratureDispatch.
 */
template <class Domain>
struct QuadratureRules;

template <class Domain, class Real>
using QuadraturePointAndWeight_t =
    std::pair<sofa::type::Vec<QuadratureRules<Domain>::TopologicalDimension, Real>, Real>;

/// Reference domain: the segment [-1, 1].
template <>
struct QuadratureRules<sofa::geometry::Edge>
{
    static constexpr sofa::Size TopologicalDimension = 1;
    static constexpr std::array<sofa::Size, 2> tabulatedDegrees { 1, 3 };

    template <sofa::Size Degree, class Real>
    static constexpr auto points()
    {
        using Point = sofa::type::Vec<TopologicalDimension, Real>;
        using PointAndWeight = std::pair<Point, Real>;

        if constexpr (Degree == 1)
        {
            // 1-point midpoint rule
            return std::array<PointAndWeight, 1>{
                std::make_pair(Point(static_cast<Real>(0)), static_cast<Real>(2))
            };
        }
        else if constexpr (Degree == 3)
        {
            // 2-point Gauss-Legendre rule
            constexpr Real g = std::numbers::inv_sqrt3_v<Real>; // 1/sqrt(3)
            return std::array<PointAndWeight, 2>{
                std::make_pair(Point(-g), static_cast<Real>(1)),
                std::make_pair(Point( g), static_cast<Real>(1))
            };
        }
        else
        {
            static_assert(degreeIsNotTabulated<Degree>, "QuadratureRules<Edge>: degree is not tabulated");
        }
    }
};

/// Reference domain: the unit triangle (0,0), (1,0), (0,1).
template <>
struct QuadratureRules<sofa::geometry::Triangle>
{
    static constexpr sofa::Size TopologicalDimension = 2;
    static constexpr std::array<sofa::Size, 2> tabulatedDegrees { 1, 2 };

    template <sofa::Size Degree, class Real>
    static constexpr auto points()
    {
        using Point = sofa::type::Vec<TopologicalDimension, Real>;
        using PointAndWeight = std::pair<Point, Real>;

        constexpr Real third = static_cast<Real>(1) / static_cast<Real>(3); // 1/3
        constexpr Real sixth = static_cast<Real>(1) / static_cast<Real>(6); // 1/6

        if constexpr (Degree == 1)
        {
            // 1-point centroid rule
            return std::array<PointAndWeight, 1>{
                std::make_pair(Point(third, third), static_cast<Real>(0.5)) // area of the reference triangle
            };
        }
        else if constexpr (Degree == 2)
        {
            // 3-point interior rule
            constexpr Real a = static_cast<Real>(2) / static_cast<Real>(3); // 2/3
            return std::array<PointAndWeight, 3>{
                std::make_pair(Point(sixth, sixth), sixth),
                std::make_pair(Point(a,     sixth), sixth),
                std::make_pair(Point(sixth, a),     sixth)
            };
        }
        else
        {
            static_assert(degreeIsNotTabulated<Degree>, "QuadratureRules<Triangle>: degree is not tabulated");
        }
    }
};

/// Reference domain: the square [-1, 1]^2.
template <>
struct QuadratureRules<sofa::geometry::Quad>
{
    static constexpr sofa::Size TopologicalDimension = 2;
    static constexpr std::array<sofa::Size, 3> tabulatedDegrees { 1, 2, 3 };

    template <sofa::Size Degree, class Real>
    static constexpr auto points()
    {
        using Point = sofa::type::Vec<TopologicalDimension, Real>;
        using PointAndWeight = std::pair<Point, Real>;

        if constexpr (Degree == 1)
        {
            // 1-point centroid rule
            return std::array<PointAndWeight, 1>{
                std::make_pair(Point(static_cast<Real>(0), static_cast<Real>(0)), static_cast<Real>(4))
            };
        }
        else if constexpr (Degree == 2)
        {
            // 3-point rule
            constexpr Real sqrt2_3 = std::numbers::sqrt2_v<Real> * std::numbers::inv_sqrt3_v<Real>; // sqrt(2/3)
            constexpr Real inv_sqrt6 = std::numbers::inv_sqrt3_v<Real> / std::numbers::sqrt2_v<Real>; // 1/sqrt(6)
            constexpr Real inv_sqrt2 = std::numbers::sqrt2_v<Real> / static_cast<Real>(2); // 1/sqrt(2)
            constexpr Real w = static_cast<Real>(4) / static_cast<Real>(3); // 4/3

            return std::array<PointAndWeight, 3>{
                std::make_pair(Point( sqrt2_3,    static_cast<Real>(0)), w),
                std::make_pair(Point(-inv_sqrt6, -inv_sqrt2),            w),
                std::make_pair(Point(-inv_sqrt6,  inv_sqrt2),            w)
            };
        }
        else if constexpr (Degree == 3)
        {
            // 2x2 Gauss-Legendre rule
            constexpr Real g = std::numbers::inv_sqrt3_v<Real>; // 1/sqrt(3)
            constexpr Real one = static_cast<Real>(1);
            return std::array<PointAndWeight, 4>{
                std::make_pair(Point(-g, -g), one),
                std::make_pair(Point( g, -g), one),
                std::make_pair(Point( g,  g), one),
                std::make_pair(Point(-g,  g), one)
            };
        }
        else
        {
            static_assert(degreeIsNotTabulated<Degree>, "QuadratureRules<Quad>: degree is not tabulated");
        }
    }
};

/// Reference domain: the unit tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1).
template <>
struct QuadratureRules<sofa::geometry::Tetrahedron>
{
    static constexpr sofa::Size TopologicalDimension = 3;
    static constexpr std::array<sofa::Size, 2> tabulatedDegrees { 1, 2 };

    template <sofa::Size Degree, class Real>
    static constexpr auto points()
    {
        using Point = sofa::type::Vec<TopologicalDimension, Real>;
        using PointAndWeight = std::pair<Point, Real>;

        if constexpr (Degree == 1)
        {
            // 1-point centroid rule
            constexpr Real quarter = static_cast<Real>(0.25); // 1/4
            return std::array<PointAndWeight, 1>{
                std::make_pair(Point(quarter, quarter, quarter),
                               static_cast<Real>(1) / static_cast<Real>(6)) // volume of the reference tetrahedron
            };
        }
        else if constexpr (Degree == 2)
        {
            // 4-point rule
            constexpr Real sqrt5 = static_cast<Real>(2.23606797749978969640917366873); // sqrt(5)
            constexpr Real a = (static_cast<Real>(5) - sqrt5) / static_cast<Real>(20);
            constexpr Real b = (static_cast<Real>(5) + static_cast<Real>(3) * sqrt5) / static_cast<Real>(20);
            constexpr Real w = static_cast<Real>(1) / static_cast<Real>(24); // 1/24

            return std::array<PointAndWeight, 4>{
                std::make_pair(Point(a, a, a), w),
                std::make_pair(Point(b, a, a), w),
                std::make_pair(Point(a, b, a), w),
                std::make_pair(Point(a, a, b), w)
            };
        }
        else
        {
            static_assert(degreeIsNotTabulated<Degree>, "QuadratureRules<Tetrahedron>: degree is not tabulated");
        }
    }
};

/// Reference domain: the cube [-1, 1]^3.
template <>
struct QuadratureRules<sofa::geometry::Hexahedron>
{
    static constexpr sofa::Size TopologicalDimension = 3;
    static constexpr std::array<sofa::Size, 3> tabulatedDegrees { 1, 3, 5 };

    template <sofa::Size Degree, class Real>
    static constexpr auto points()
    {
        using Point = sofa::type::Vec<TopologicalDimension, Real>;
        using PointAndWeight = std::pair<Point, Real>;

        if constexpr (Degree == 1)
        {
            // 1-point centroid rule
            constexpr Real zero = static_cast<Real>(0);
            return std::array<PointAndWeight, 1>{
                std::make_pair(Point(zero, zero, zero), static_cast<Real>(8))
            };
        }
        else if constexpr (Degree == 3)
        {
            // 2x2x2 Gauss-Legendre rule, in the node order of sofa::geometry::Hexahedron
            constexpr Real g = std::numbers::inv_sqrt3_v<Real>; // 1/sqrt(3)
            constexpr Real one = static_cast<Real>(1);
            return std::array<PointAndWeight, 8>{
                std::make_pair(Point(-g, -g, -g), one),
                std::make_pair(Point( g, -g, -g), one),
                std::make_pair(Point( g,  g, -g), one),
                std::make_pair(Point(-g,  g, -g), one),
                std::make_pair(Point(-g, -g,  g), one),
                std::make_pair(Point( g, -g,  g), one),
                std::make_pair(Point( g,  g,  g), one),
                std::make_pair(Point(-g,  g,  g), one)
            };
        }
        else if constexpr (Degree == 5)
        {
            // 3x3x3 Gauss-Legendre rule
            constexpr Real g = static_cast<Real>(0.774596669241483377035853079956); // sqrt(3/5)
            constexpr std::array<Real, 3> node { -g, static_cast<Real>(0), g };
            constexpr std::array<Real, 3> weight {
                static_cast<Real>(5) / static_cast<Real>(9), // 5/9
                static_cast<Real>(8) / static_cast<Real>(9), // 8/9
                static_cast<Real>(5) / static_cast<Real>(9)  // 5/9
            };

            std::array<PointAndWeight, 27> q {};
            sofa::Size k = 0;
            for (sofa::Size i = 0; i < 3; ++i)
                for (sofa::Size j = 0; j < 3; ++j)
                    for (sofa::Size l = 0; l < 3; ++l)
                        q[k++] = std::make_pair(Point(node[i], node[j], node[l]),
                                                weight[i] * weight[j] * weight[l]);
            return q;
        }
        else
        {
            static_assert(degreeIsNotTabulated<Degree>, "QuadratureRules<Hexahedron>: degree is not tabulated");
        }
    }
};

/// Reference domain: the unit triangle extruded along z over [0, 1].
template <>
struct QuadratureRules<sofa::geometry::Prism>
{
    static constexpr sofa::Size TopologicalDimension = 3;
    static constexpr std::array<sofa::Size, 1> tabulatedDegrees { 1 };

    template <sofa::Size Degree, class Real>
    static constexpr auto points()
    {
        using Point = sofa::type::Vec<TopologicalDimension, Real>;
        using PointAndWeight = std::pair<Point, Real>;

        if constexpr (Degree == 1)
        {
            // centroid of the triangle x 2-point Gauss-Legendre along the extrusion
            constexpr Real third = static_cast<Real>(1) / static_cast<Real>(3); // 1/3
            constexpr Real g = std::numbers::inv_sqrt3_v<Real>; // 1/sqrt(3)
            constexpr Real half = static_cast<Real>(0.5);
            constexpr Real one = static_cast<Real>(1);
            constexpr Real w = static_cast<Real>(0.25); // 1/4

            return std::array<PointAndWeight, 2>{
                std::make_pair(Point(third, third, half * (one - g)), w),
                std::make_pair(Point(third, third, half * (one + g)), w)
            };
        }
        else
        {
            static_assert(degreeIsNotTabulated<Degree>, "QuadratureRules<Prism>: degree is not tabulated");
        }
    }
};

/// Reference domain: the cube [-1, 1]^3 collapsed onto the pyramid apex.
template <>
struct QuadratureRules<sofa::geometry::Pyramid>
{
    static constexpr sofa::Size TopologicalDimension = 3;
    static constexpr std::array<sofa::Size, 1> tabulatedDegrees { 1 };

    template <sofa::Size Degree, class Real>
    static constexpr auto points()
    {
        using Point = sofa::type::Vec<TopologicalDimension, Real>;
        using PointAndWeight = std::pair<Point, Real>;

        if constexpr (Degree == 1)
        {
            // 2x2x2 Gauss-Legendre rule
            constexpr Real g = std::numbers::inv_sqrt3_v<Real>; // 1/sqrt(3)
            constexpr Real one = static_cast<Real>(1);
            return std::array<PointAndWeight, 8>{
                std::make_pair(Point(-g, -g, -g), one),
                std::make_pair(Point( g, -g, -g), one),
                std::make_pair(Point( g,  g, -g), one),
                std::make_pair(Point(-g,  g, -g), one),
                std::make_pair(Point(-g, -g,  g), one),
                std::make_pair(Point( g, -g,  g), one),
                std::make_pair(Point( g,  g,  g), one),
                std::make_pair(Point(-g,  g,  g), one)
            };
        }
        else
        {
            static_assert(degreeIsNotTabulated<Degree>, "QuadratureRules<Pyramid>: degree is not tabulated");
        }
    }
};

/**
 * Selects a quadrature rule out of QuadratureRules<Domain>, at compile time or at runtime.
 *
 * A requested degree that is not tabulated is served by the smallest tabulated degree
 * above it, so a caller states the accuracy it needs and never a table size. The
 * selection is derived from Rules::tabulatedDegrees, which stays the single place where
 * the available rules of a domain are listed.
 */
template <class Domain>
struct QuadratureDispatch
{
    using Rules = QuadratureRules<Domain>;
    static constexpr sofa::Size TopologicalDimension = Rules::TopologicalDimension;

    template <class Real>
    using PointAndWeight = QuadraturePointAndWeight_t<Domain, Real>;

    /// Smallest tabulated degree integrating a polynomial of the requested degree
    /// exactly, or 0 when the request exceeds every table of this domain.
    static constexpr sofa::Size smallestSufficientDegree(sofa::Size requestedDegree)
    {
        for (const sofa::Size degree : Rules::tabulatedDegrees)
        {
            if (degree >= requestedDegree)
            {
                return degree;
            }
        }
        return 0;
    }

    /// The table itself, selected at compile time.
    template <sofa::Size RequestedDegree, class Real>
    static constexpr auto points()
    {
        constexpr sofa::Size degree = smallestSufficientDegree(RequestedDegree);
        static_assert(degree != 0,
            "QuadratureDispatch: no quadrature rule is tabulated for the requested degree");
        return Rules::template points<degree, Real>();
    }

    /// View of the table matching a degree known only at runtime.
    template <class Real>
    static std::span<const PointAndWeight<Real>> rule(sofa::Size requestedDegree)
    {
        return select<Real>(requestedDegree,
            std::make_index_sequence<Rules::tabulatedDegrees.size()>{});
    }

private:
    /// The table of one tabulated degree.
    template <sofa::Size Degree, class Real>
    static std::span<const PointAndWeight<Real>> table()
    {
        static constexpr auto tabulated = Rules::template points<Degree, Real>();
        return tabulated;
    }

    /**
     * Runtime counterpart of smallestSufficientDegree: keeps the table of the first
     * tabulated degree that covers the request.
     *
     * Index is the pack 0, 1, ... over Rules::tabulatedDegrees, so the fold below expands
     * to one test per tabulated degree. For a domain tabulating {1, 3, 5} it reads:
     *
     *     (requestedDegree <= 1 ? (selected = table<1, Real>(), true) : false) ||
     *     (requestedDegree <= 3 ? (selected = table<3, Real>(), true) : false) ||
     *     (requestedDegree <= 5 ? (selected = table<5, Real>(), true) : false)
     *
     * tabulatedDegrees is ascending and || short-circuits, so the first sufficient table
     * is the one kept and the smaller ones are never evaluated. Each term yields a bool
     * because a fold over || demands one; the comma operator performs the assignment and
     * then reports true. The fold is false only when no degree matched, hence the throw.
     */
    template <class Real, std::size_t... Index>
    static std::span<const PointAndWeight<Real>> select(
        sofa::Size requestedDegree, std::index_sequence<Index...>)
    {
        std::span<const PointAndWeight<Real>> selected;

        const bool isTabulated = ((requestedDegree <= Rules::tabulatedDegrees[Index]
            ? (selected = table<Rules::tabulatedDegrees[Index], Real>(), true)
            : false) || ...);

        if (!isTabulated)
        {
            throw std::invalid_argument(
                std::string("QuadratureDispatch<")
                + sofa::geometry::elementTypeToString(Domain::Element_type)
                + ">: no quadrature rule is tabulated for degree "
                + std::to_string(requestedDegree));
        }

        return selected;
    }
};

/**
 * Quadrature interface of a finite element: the rules of its reference domain, restricted
 * to the degrees its interpolation order admits.
 *
 * ElementType supplies the domain (through ReferenceDomain) and the interpolation order
 * (through sofa::geometry), so an element only states the degree it wants by default.
 * DefaultDegree is given explicitly rather than derived, because the accuracy an element
 * ships with is a choice, not a consequence of its order.
 */
template <class ElementType, class Real, sofa::Size DefaultDegree>
struct FiniteElementQuadrature
{
    using Domain = ReferenceDomain_t<ElementType>;
    using Dispatch = QuadratureDispatch<Domain>;
    using QuadraturePointAndWeight = QuadraturePointAndWeight_t<Domain, Real>;

    static constexpr sofa::Size InterpolationOrder = ElementType::PolynomialOrder;

    /// Lowest degree integrating the stiffness integrand of an order-p element exactly on
    /// an affine mapping: the product of two shape function gradients is of degree 2(p-1).
    /// Clamped to 1, since no degree-0 rule is tabulated.
    static constexpr sofa::Size MinimumQuadratureDegree =
        InterpolationOrder > 1 ? 2 * (InterpolationOrder - 1) : 1;

    static constexpr sofa::Size DefaultQuadratureDegree = DefaultDegree;
    static_assert(DefaultQuadratureDegree >= MinimumQuadratureDegree,
        "the default quadrature degree is too low for the order of this element");

    /// Quadrature points and weights of the given degree, resolved at compile time.
    template <sofa::Size Degree = DefaultQuadratureDegree>
    static constexpr auto quadraturePoints()
    {
        static_assert(Degree >= MinimumQuadratureDegree,
            "quadrature degree too low for the interpolation order of this element");
        return Dispatch::template points<Degree, Real>();
    }

    /// Quadrature points and weights of a degree known only at runtime; view of the
    /// compile-time table.
    static std::span<const QuadraturePointAndWeight> quadratureRule(sofa::Size degree)
    {
        if (degree < MinimumQuadratureDegree)
        {
            throw std::invalid_argument(
                std::string("FiniteElement<")
                + sofa::geometry::elementTypeToString(ElementType::Element_type)
                + ">::quadratureRule: degree " + std::to_string(degree)
                + " is too low for an element of interpolation order "
                + std::to_string(InterpolationOrder));
        }
        return Dispatch::template rule<Real>(degree);
    }
};

}
