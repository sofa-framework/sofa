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
#include <sofa/fem/FiniteElement.h>

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_PYRAMID_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::QuadraticPyramid, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::QuadraticPyramid, DataTypes, 3);
    static_assert(spatial_dimensions == 3, "Quadratic Pyramids are only defined in 3D");

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        // 5 corner vertices
        {-1, -1, 0},   // vertex 0
        {1, -1, 0},    // vertex 1
        {1, 1, 0},     // vertex 2
        {-1, 1, 0},    // vertex 3
        {0, 0, 1},     // vertex 4 (apex)
        // 8 mid-edge nodes
        {0, -1, 0},    // mid-edge 0-1
        {1, 0, 0},     // mid-edge 1-2
        {0, 1, 0},     // mid-edge 2-3
        {-1, 0, 0},    // mid-edge 3-0
        {-0.5, -0.5, 0.5},  // mid-edge 0-4
        {0.5, -0.5, 0.5},   // mid-edge 1-4
        {0.5, 0.5, 0.5},    // mid-edge 2-4
        {-0.5, 0.5, 0.5},   // mid-edge 3-4
        // 1 face-center node (on the quad base)
        {0, 0, 0}      // face center of base quad
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getElements<sofa::geometry::QuadraticPyramid>();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real xi = q[0];
        const Real eta = q[1];
        const Real zeta = q[2];
        const Real r = (zeta < 0.999) ? (1 - zeta) : 0.001; // Avoid singularity at apex

        // Corner vertices (5)
        const Real N0 = 0.125 * (xi * xi - xi) * (eta * eta - eta) / r;
        const Real N1 = 0.125 * (xi * xi + xi) * (eta * eta - eta) / r;
        const Real N2 = 0.125 * (xi * xi + xi) * (eta * eta + eta) / r;
        const Real N3 = 0.125 * (xi * xi - xi) * (eta * eta + eta) / r;
        const Real N4 = zeta * (2 * zeta - 1);

        // Mid-edge nodes on base (4)
        const Real N5 = 0.25 * (1 - xi * xi) * (eta * eta - eta) / r;
        const Real N6 = 0.25 * (xi * xi + xi) * (1 - eta * eta) / r;
        const Real N7 = 0.25 * (1 - xi * xi) * (eta * eta + eta) / r;
        const Real N8 = 0.25 * (xi * xi - xi) * (1 - eta * eta) / r;

        // Mid-edge nodes connecting to apex (4)
        const Real N9 = 0.5 * (xi * xi - xi) * (eta * eta - eta) * zeta / r;
        const Real N10 = 0.5 * (xi * xi + xi) * (eta * eta - eta) * zeta / r;
        const Real N11 = 0.5 * (xi * xi + xi) * (eta * eta + eta) * zeta / r;
        const Real N12 = 0.5 * (xi * xi - xi) * (eta * eta + eta) * zeta / r;

        // Face-center node on base (1)
        const Real N13 = 0.5 * (1 - xi * xi) * (1 - eta * eta) / r;

        return {N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13};
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real xi = q[0];
        const Real eta = q[1];
        const Real zeta = q[2];
        const Real r = (zeta < 0.999) ? (1 - zeta) : 0.001;

        return {
            // Corner vertices (5)
            {0.125 * (2 * xi - 1) * (eta * eta - eta) / r,
             0.125 * (xi * xi - xi) * (2 * eta - 1) / r,
             0.125 * (xi * xi - xi) * (eta * eta - eta) / (r * r)},

            {0.125 * (2 * xi + 1) * (eta * eta - eta) / r,
             0.125 * (xi * xi + xi) * (2 * eta - 1) / r,
             0.125 * (xi * xi + xi) * (eta * eta - eta) / (r * r)},

            {0.125 * (2 * xi + 1) * (eta * eta + eta) / r,
             0.125 * (xi * xi + xi) * (2 * eta + 1) / r,
             0.125 * (xi * xi + xi) * (eta * eta + eta) / (r * r)},

            {0.125 * (2 * xi - 1) * (eta * eta + eta) / r,
             0.125 * (xi * xi - xi) * (2 * eta + 1) / r,
             0.125 * (xi * xi - xi) * (eta * eta + eta) / (r * r)},

            {0, 0, 4 * zeta - 1},

            // Mid-edge nodes on base (4)
            {-0.5 * xi * (eta * eta - eta) / r,
             0.25 * (1 - xi * xi) * (2 * eta - 1) / r,
             0.25 * (1 - xi * xi) * (eta * eta - eta) / (r * r)},

            {0.25 * (2 * xi + 1) * (1 - eta * eta) / r,
             -0.5 * (xi * xi + xi) * eta / r,
             0.25 * (xi * xi + xi) * (1 - eta * eta) / (r * r)},

            {-0.5 * xi * (eta * eta + eta) / r,
             0.25 * (1 - xi * xi) * (2 * eta + 1) / r,
             0.25 * (1 - xi * xi) * (eta * eta + eta) / (r * r)},

            {0.25 * (2 * xi - 1) * (1 - eta * eta) / r,
             -0.5 * (xi * xi - xi) * eta / r,
             0.25 * (xi * xi - xi) * (1 - eta * eta) / (r * r)},

            // Mid-edge nodes connecting to apex (4)
            {0.5 * (2 * xi - 1) * (eta * eta - eta) * zeta / r,
             0.5 * (xi * xi - xi) * (2 * eta - 1) * zeta / r,
             0.5 * (xi * xi - xi) * (eta * eta - eta) * (r - zeta) / (r * r)},

            {0.5 * (2 * xi + 1) * (eta * eta - eta) * zeta / r,
             0.5 * (xi * xi + xi) * (2 * eta - 1) * zeta / r,
             0.5 * (xi * xi + xi) * (eta * eta - eta) * (r - zeta) / (r * r)},

            {0.5 * (2 * xi + 1) * (eta * eta + eta) * zeta / r,
             0.5 * (xi * xi + xi) * (2 * eta + 1) * zeta / r,
             0.5 * (xi * xi + xi) * (eta * eta + eta) * (r - zeta) / (r * r)},

            {0.5 * (2 * xi - 1) * (eta * eta + eta) * zeta / r,
             0.5 * (xi * xi - xi) * (2 * eta + 1) * zeta / r,
             0.5 * (xi * xi - xi) * (eta * eta + eta) * (r - zeta) / (r * r)},

            // Face-center node on base (1)
            {-xi * (1 - eta * eta) / r,
             -eta * (1 - xi * xi) / r,
             0.5 * (1 - xi * xi) * (1 - eta * eta) / (r * r)}
        };
    }

    static constexpr std::array<QuadraturePointAndWeight, 5> quadraturePoints()
    {
        // 5-point quadrature for pyramid
        constexpr Real a = 0.584237394672974;
        constexpr Real b = 0.138196601125011;
        constexpr Real w1 = 0.133333333333333;
        constexpr Real w2 = 0.075000000000000;

        constexpr sofa::type::Vec<TopologicalDimension, Real> q0(0, 0, b);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q1(a, 0, 0.5);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q2(0, a, 0.5);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q3(-a, 0, 0.5);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q4(0, -a, 0.5);

        constexpr std::array<QuadraturePointAndWeight, 5> q {
            std::make_pair(q0, w1),
            std::make_pair(q1, w2),
            std::make_pair(q2, w2),
            std::make_pair(q3, w2),
            std::make_pair(q4, w2)
        };
        return q;
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_PYRAMID_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticPyramid, sofa::defaulttype::Vec3Types>;
#endif

}
