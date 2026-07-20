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

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_HEXAHEDRON_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::QuadraticHexahedron, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::QuadraticHexahedron, DataTypes, 3);
    static_assert(spatial_dimensions == 3, "Quadratic Hexahedrons are only defined in 3D");

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        // 8 corner vertices
        {-1, -1, -1},  // vertex 0
        {1, -1, -1},   // vertex 1
        {1, 1, -1},    // vertex 2
        {-1, 1, -1},   // vertex 3
        {-1, -1, 1},   // vertex 4
        {1, -1, 1},    // vertex 5
        {1, 1, 1},     // vertex 6
        {-1, 1, 1},    // vertex 7
        // 12 mid-edge nodes
        {0, -1, -1},   // mid-edge 0-1
        {1, 0, -1},    // mid-edge 1-2
        {0, 1, -1},    // mid-edge 2-3
        {-1, 0, -1},   // mid-edge 3-0
        {-1, -1, 0},   // mid-edge 0-4
        {1, -1, 0},    // mid-edge 1-5
        {1, 1, 0},     // mid-edge 2-6
        {-1, 1, 0},    // mid-edge 3-7
        {0, -1, 1},    // mid-edge 4-5
        {1, 0, 1},     // mid-edge 5-6
        {0, 1, 1},     // mid-edge 6-7
        {-1, 0, 1},    // mid-edge 7-4
        // 6 face-center nodes
        {0, 0, -1},    // face center bottom (0-1-2-3)
        {0, -1, 0},    // face center front (0-1-5-4)
        {1, 0, 0},     // face center right (1-2-6-5)
        {0, 1, 0},     // face center back (2-3-7-6)
        {-1, 0, 0},    // face center left (3-0-4-7)
        {0, 0, 1},     // face center top (4-5-6-7)
        // 1 volume center node
        {0, 0, 0}      // volume center
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getElements<sofa::geometry::QuadraticHexahedron>();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real xi = q[0];
        const Real eta = q[1];
        const Real zeta = q[2];

        // Corner nodes (8)
        const Real N0 = 0.125 * (xi * xi - xi) * (eta * eta - eta) * (zeta * zeta - zeta);
        const Real N1 = 0.125 * (xi * xi + xi) * (eta * eta - eta) * (zeta * zeta - zeta);
        const Real N2 = 0.125 * (xi * xi + xi) * (eta * eta + eta) * (zeta * zeta - zeta);
        const Real N3 = 0.125 * (xi * xi - xi) * (eta * eta + eta) * (zeta * zeta - zeta);
        const Real N4 = 0.125 * (xi * xi - xi) * (eta * eta - eta) * (zeta * zeta + zeta);
        const Real N5 = 0.125 * (xi * xi + xi) * (eta * eta - eta) * (zeta * zeta + zeta);
        const Real N6 = 0.125 * (xi * xi + xi) * (eta * eta + eta) * (zeta * zeta + zeta);
        const Real N7 = 0.125 * (xi * xi - xi) * (eta * eta + eta) * (zeta * zeta + zeta);

        // Mid-edge nodes (12)
        const Real N8 = 0.25 * (1 - xi * xi) * (eta * eta - eta) * (zeta * zeta - zeta);
        const Real N9 = 0.25 * (xi * xi + xi) * (1 - eta * eta) * (zeta * zeta - zeta);
        const Real N10 = 0.25 * (1 - xi * xi) * (eta * eta + eta) * (zeta * zeta - zeta);
        const Real N11 = 0.25 * (xi * xi - xi) * (1 - eta * eta) * (zeta * zeta - zeta);
        const Real N12 = 0.25 * (xi * xi - xi) * (eta * eta - eta) * (1 - zeta * zeta);
        const Real N13 = 0.25 * (xi * xi + xi) * (eta * eta - eta) * (1 - zeta * zeta);
        const Real N14 = 0.25 * (xi * xi + xi) * (eta * eta + eta) * (1 - zeta * zeta);
        const Real N15 = 0.25 * (xi * xi - xi) * (eta * eta + eta) * (1 - zeta * zeta);
        const Real N16 = 0.25 * (1 - xi * xi) * (eta * eta - eta) * (zeta * zeta + zeta);
        const Real N17 = 0.25 * (xi * xi + xi) * (1 - eta * eta) * (zeta * zeta + zeta);
        const Real N18 = 0.25 * (1 - xi * xi) * (eta * eta + eta) * (zeta * zeta + zeta);
        const Real N19 = 0.25 * (xi * xi - xi) * (1 - eta * eta) * (zeta * zeta + zeta);

        // Face-center nodes (6)
        const Real N20 = 0.5 * (1 - xi * xi) * (1 - eta * eta) * (zeta * zeta - zeta);
        const Real N21 = 0.5 * (1 - xi * xi) * (eta * eta - eta) * (1 - zeta * zeta);
        const Real N22 = 0.5 * (xi * xi + xi) * (1 - eta * eta) * (1 - zeta * zeta);
        const Real N23 = 0.5 * (1 - xi * xi) * (eta * eta + eta) * (1 - zeta * zeta);
        const Real N24 = 0.5 * (xi * xi - xi) * (1 - eta * eta) * (1 - zeta * zeta);
        const Real N25 = 0.5 * (1 - xi * xi) * (1 - eta * eta) * (zeta * zeta + zeta);

        // Volume-center node (1)
        const Real N26 = (1 - xi * xi) * (1 - eta * eta) * (1 - zeta * zeta);

        return {N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17, N18, N19, N20, N21, N22, N23, N24, N25, N26};
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real xi = q[0];
        const Real eta = q[1];
        const Real zeta = q[2];

        return {
            // Corner vertices (8)
            {0.125 * (2 * xi - 1) * (eta * eta - eta) * (zeta * zeta - zeta),
             0.125 * (xi * xi - xi) * (2 * eta - 1) * (zeta * zeta - zeta),
             0.125 * (xi * xi - xi) * (eta * eta - eta) * (2 * zeta - 1)},

            {0.125 * (2 * xi + 1) * (eta * eta - eta) * (zeta * zeta - zeta),
             0.125 * (xi * xi + xi) * (2 * eta - 1) * (zeta * zeta - zeta),
             0.125 * (xi * xi + xi) * (eta * eta - eta) * (2 * zeta - 1)},

            {0.125 * (2 * xi + 1) * (eta * eta + eta) * (zeta * zeta - zeta),
             0.125 * (xi * xi + xi) * (2 * eta + 1) * (zeta * zeta - zeta),
             0.125 * (xi * xi + xi) * (eta * eta + eta) * (2 * zeta - 1)},

            {0.125 * (2 * xi - 1) * (eta * eta + eta) * (zeta * zeta - zeta),
             0.125 * (xi * xi - xi) * (2 * eta + 1) * (zeta * zeta - zeta),
             0.125 * (xi * xi - xi) * (eta * eta + eta) * (2 * zeta - 1)},

            {0.125 * (2 * xi - 1) * (eta * eta - eta) * (zeta * zeta + zeta),
             0.125 * (xi * xi - xi) * (2 * eta - 1) * (zeta * zeta + zeta),
             0.125 * (xi * xi - xi) * (eta * eta - eta) * (2 * zeta + 1)},

            {0.125 * (2 * xi + 1) * (eta * eta - eta) * (zeta * zeta + zeta),
             0.125 * (xi * xi + xi) * (2 * eta - 1) * (zeta * zeta + zeta),
             0.125 * (xi * xi + xi) * (eta * eta - eta) * (2 * zeta + 1)},

            {0.125 * (2 * xi + 1) * (eta * eta + eta) * (zeta * zeta + zeta),
             0.125 * (xi * xi + xi) * (2 * eta + 1) * (zeta * zeta + zeta),
             0.125 * (xi * xi + xi) * (eta * eta + eta) * (2 * zeta + 1)},

            {0.125 * (2 * xi - 1) * (eta * eta + eta) * (zeta * zeta + zeta),
             0.125 * (xi * xi - xi) * (2 * eta + 1) * (zeta * zeta + zeta),
             0.125 * (xi * xi - xi) * (eta * eta + eta) * (2 * zeta + 1)},

            // Mid-edge nodes (12)
            {-0.5 * xi * (eta * eta - eta) * (zeta * zeta - zeta),
             0.25 * (1 - xi * xi) * (2 * eta - 1) * (zeta * zeta - zeta),
             0.25 * (1 - xi * xi) * (eta * eta - eta) * (2 * zeta - 1)},

            {0.25 * (2 * xi + 1) * (1 - eta * eta) * (zeta * zeta - zeta),
             -0.5 * (xi * xi + xi) * eta * (zeta * zeta - zeta),
             0.25 * (xi * xi + xi) * (1 - eta * eta) * (2 * zeta - 1)},

            {-0.5 * xi * (eta * eta + eta) * (zeta * zeta - zeta),
             0.25 * (1 - xi * xi) * (2 * eta + 1) * (zeta * zeta - zeta),
             0.25 * (1 - xi * xi) * (eta * eta + eta) * (2 * zeta - 1)},

            {0.25 * (2 * xi - 1) * (1 - eta * eta) * (zeta * zeta - zeta),
             -0.5 * (xi * xi - xi) * eta * (zeta * zeta - zeta),
             0.25 * (xi * xi - xi) * (1 - eta * eta) * (2 * zeta - 1)},

            {0.25 * (2 * xi - 1) * (eta * eta - eta) * (1 - zeta * zeta),
             0.25 * (xi * xi - xi) * (2 * eta - 1) * (1 - zeta * zeta),
             -0.5 * (xi * xi - xi) * (eta * eta - eta) * zeta},

            {0.25 * (2 * xi + 1) * (eta * eta - eta) * (1 - zeta * zeta),
             0.25 * (xi * xi + xi) * (2 * eta - 1) * (1 - zeta * zeta),
             -0.5 * (xi * xi + xi) * (eta * eta - eta) * zeta},

            {0.25 * (2 * xi + 1) * (eta * eta + eta) * (1 - zeta * zeta),
             0.25 * (xi * xi + xi) * (2 * eta + 1) * (1 - zeta * zeta),
             -0.5 * (xi * xi + xi) * (eta * eta + eta) * zeta},

            {0.25 * (2 * xi - 1) * (eta * eta + eta) * (1 - zeta * zeta),
             0.25 * (xi * xi - xi) * (2 * eta + 1) * (1 - zeta * zeta),
             -0.5 * (xi * xi - xi) * (eta * eta + eta) * zeta},

            {-0.5 * xi * (eta * eta - eta) * (zeta * zeta + zeta),
             0.25 * (1 - xi * xi) * (2 * eta - 1) * (zeta * zeta + zeta),
             0.25 * (1 - xi * xi) * (eta * eta - eta) * (2 * zeta + 1)},

            {0.25 * (2 * xi + 1) * (1 - eta * eta) * (zeta * zeta + zeta),
             -0.5 * (xi * xi + xi) * eta * (zeta * zeta + zeta),
             0.25 * (xi * xi + xi) * (1 - eta * eta) * (2 * zeta + 1)},

            {-0.5 * xi * (eta * eta + eta) * (zeta * zeta + zeta),
             0.25 * (1 - xi * xi) * (2 * eta + 1) * (zeta * zeta + zeta),
             0.25 * (1 - xi * xi) * (eta * eta + eta) * (2 * zeta + 1)},

            {0.25 * (2 * xi - 1) * (1 - eta * eta) * (zeta * zeta + zeta),
             -0.5 * (xi * xi - xi) * eta * (zeta * zeta + zeta),
             0.25 * (xi * xi - xi) * (1 - eta * eta) * (2 * zeta + 1)},

            // Face-center nodes (6)
            {-xi * (1 - eta * eta) * (zeta * zeta - zeta),
             -eta * (1 - xi * xi) * (zeta * zeta - zeta),
             0.5 * (1 - xi * xi) * (1 - eta * eta) * (2 * zeta - 1)},

            {-xi * (eta * eta - eta) * (1 - zeta * zeta),
             0.5 * (1 - xi * xi) * (2 * eta - 1) * (1 - zeta * zeta),
             -zeta * (1 - xi * xi) * (eta * eta - eta)},

            {0.5 * (2 * xi + 1) * (1 - eta * eta) * (1 - zeta * zeta),
             -eta * (xi * xi + xi) * (1 - zeta * zeta),
             -zeta * (xi * xi + xi) * (1 - eta * eta)},

            {-xi * (eta * eta + eta) * (1 - zeta * zeta),
             0.5 * (1 - xi * xi) * (2 * eta + 1) * (1 - zeta * zeta),
             -zeta * (1 - xi * xi) * (eta * eta + eta)},

            {0.5 * (2 * xi - 1) * (1 - eta * eta) * (1 - zeta * zeta),
             -eta * (xi * xi - xi) * (1 - zeta * zeta),
             -zeta * (xi * xi - xi) * (1 - eta * eta)},

            {-xi * (1 - eta * eta) * (zeta * zeta + zeta),
             -eta * (1 - xi * xi) * (zeta * zeta + zeta),
             0.5 * (1 - xi * xi) * (1 - eta * eta) * (2 * zeta + 1)},

            // Volume-center node (1)
            {-2 * xi * (1 - eta * eta) * (1 - zeta * zeta),
             -2 * eta * (1 - xi * xi) * (1 - zeta * zeta),
             -2 * zeta * (1 - xi * xi) * (1 - eta * eta)}
        };
    }

    static constexpr std::array<QuadraturePointAndWeight, 8> quadraturePoints()
    {
        // constexpr Real a = 1. / std::sqrt(3.);
        constexpr Real a { 0.57735026919 };
        constexpr Real w = 1.0;

        constexpr sofa::type::Vec<TopologicalDimension, Real> q0(-a, -a, -a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q1(a, -a, -a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q2(a, a, -a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q3(-a, a, -a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q4(-a, -a, a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q5(a, -a, a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q6(a, a, a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q7(-a, a, a);

        constexpr std::array<QuadraturePointAndWeight, 8> q {
            std::make_pair(q0, w),
            std::make_pair(q1, w),
            std::make_pair(q2, w),
            std::make_pair(q3, w),
            std::make_pair(q4, w),
            std::make_pair(q5, w),
            std::make_pair(q6, w),
            std::make_pair(q7, w)
        };
        return q;
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_HEXAHEDRON_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticHexahedron, sofa::defaulttype::Vec3Types>;
#endif

}
