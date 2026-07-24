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

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_PRISM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::QuadraticPrism, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::QuadraticPrism, DataTypes, 3);
    static_assert(spatial_dimensions == 3, "Quadratic Prisms are only defined in 3D");

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        // 6 corner vertices
        {0, 0, -1},      // vertex 0
        {1, 0, -1},      // vertex 1
        {0, 1, -1},      // vertex 2
        {0, 0, 1},       // vertex 3
        {1, 0, 1},       // vertex 4
        {0, 1, 1},       // vertex 5
        // 9 mid-edge nodes
        {0.5, 0, -1},    // mid-edge 0-1
        {0.5, 0.5, -1},  // mid-edge 1-2
        {0, 0.5, -1},    // mid-edge 2-0
        {0, 0, 0},       // mid-edge 0-3
        {1, 0, 0},       // mid-edge 1-4
        {0, 1, 0},       // mid-edge 2-5
        {0.5, 0, 1},     // mid-edge 3-4
        {0.5, 0.5, 1},   // mid-edge 4-5
        {0, 0.5, 1},     // mid-edge 5-3
        // 3 face-center nodes (for the 3 quad faces)
        {0.5, 0, 0},     // face center (0-1-4-3)
        {0.5, 0.5, 0},   // face center (1-2-5-4)
        {0, 0.5, 0}      // face center (2-0-3-5)
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getElements<sofa::geometry::QuadraticPrism>();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real l0 = static_cast<Real>(1) - q[0] - q[1];
        const Real l1 = q[0];
        const Real l2 = q[1];
        const Real zeta = q[2];

        // Corner vertices (6)
        const Real N0 = 0.125 * l0 * (2 * l0 - 1) * (zeta * zeta - zeta);
        const Real N1 = 0.125 * l1 * (2 * l1 - 1) * (zeta * zeta - zeta);
        const Real N2 = 0.125 * l2 * (2 * l2 - 1) * (zeta * zeta - zeta);
        const Real N3 = 0.125 * l0 * (2 * l0 - 1) * (zeta * zeta + zeta);
        const Real N4 = 0.125 * l1 * (2 * l1 - 1) * (zeta * zeta + zeta);
        const Real N5 = 0.125 * l2 * (2 * l2 - 1) * (zeta * zeta + zeta);

        // Mid-edge nodes on triangular faces (6)
        const Real N6 = 0.5 * l0 * l1 * (zeta * zeta - zeta);
        const Real N7 = 0.5 * l1 * l2 * (zeta * zeta - zeta);
        const Real N8 = 0.5 * l2 * l0 * (zeta * zeta - zeta);
        const Real N12 = 0.5 * l0 * l1 * (zeta * zeta + zeta);
        const Real N13 = 0.5 * l1 * l2 * (zeta * zeta + zeta);
        const Real N14 = 0.5 * l2 * l0 * (zeta * zeta + zeta);

        // Mid-edge nodes connecting triangular faces (3)
        const Real N9 = 0.25 * l0 * (2 * l0 - 1) * (1 - zeta * zeta);
        const Real N10 = 0.25 * l1 * (2 * l1 - 1) * (1 - zeta * zeta);
        const Real N11 = 0.25 * l2 * (2 * l2 - 1) * (1 - zeta * zeta);

        // Face-center nodes on quad faces (3)
        const Real N15 = 0.5 * l0 * l1 * (1 - zeta * zeta);
        const Real N16 = 0.5 * l1 * l2 * (1 - zeta * zeta);
        const Real N17 = 0.5 * l2 * l0 * (1 - zeta * zeta);

        return {N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17};
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real l0 = static_cast<Real>(1) - q[0] - q[1];
        const Real l1 = q[0];
        const Real l2 = q[1];
        const Real zeta = q[2];

        return {
            // Corner vertices (6)
            {0.125 * (-4 * l0 + 1) * (zeta * zeta - zeta),
             0.125 * (-4 * l0 + 1) * (zeta * zeta - zeta),
             0.125 * l0 * (2 * l0 - 1) * (2 * zeta - 1)},

            {0.125 * (4 * l1 - 1) * (zeta * zeta - zeta),
             0,
             0.125 * l1 * (2 * l1 - 1) * (2 * zeta - 1)},

            {0,
             0.125 * (4 * l2 - 1) * (zeta * zeta - zeta),
             0.125 * l2 * (2 * l2 - 1) * (2 * zeta - 1)},

            {0.125 * (-4 * l0 + 1) * (zeta * zeta + zeta),
             0.125 * (-4 * l0 + 1) * (zeta * zeta + zeta),
             0.125 * l0 * (2 * l0 - 1) * (2 * zeta + 1)},

            {0.125 * (4 * l1 - 1) * (zeta * zeta + zeta),
             0,
             0.125 * l1 * (2 * l1 - 1) * (2 * zeta + 1)},

            {0,
             0.125 * (4 * l2 - 1) * (zeta * zeta + zeta),
             0.125 * l2 * (2 * l2 - 1) * (2 * zeta + 1)},

            // Mid-edge nodes on triangular faces (6)
            {0.5 * (l1 - l0) * (zeta * zeta - zeta),
             -0.5 * l1 * (zeta * zeta - zeta),
             0.5 * l0 * l1 * (2 * zeta - 1)},

            {0.5 * l2 * (zeta * zeta - zeta),
             0.5 * l1 * (zeta * zeta - zeta),
             0.5 * l1 * l2 * (2 * zeta - 1)},

            {-0.5 * l2 * (zeta * zeta - zeta),
             0.5 * (l0 - l2) * (zeta * zeta - zeta),
             0.5 * l2 * l0 * (2 * zeta - 1)},

            {0.25 * (-4 * l0 + 1) * (1 - zeta * zeta),
             0.25 * (-4 * l0 + 1) * (1 - zeta * zeta),
             -0.5 * l0 * (2 * l0 - 1) * zeta},

            {0.25 * (4 * l1 - 1) * (1 - zeta * zeta),
             0,
             -0.5 * l1 * (2 * l1 - 1) * zeta},

            {0,
             0.25 * (4 * l2 - 1) * (1 - zeta * zeta),
             -0.5 * l2 * (2 * l2 - 1) * zeta},

            {0.5 * (l1 - l0) * (zeta * zeta + zeta),
             -0.5 * l1 * (zeta * zeta + zeta),
             0.5 * l0 * l1 * (2 * zeta + 1)},

            {0.5 * l2 * (zeta * zeta + zeta),
             0.5 * l1 * (zeta * zeta + zeta),
             0.5 * l1 * l2 * (2 * zeta + 1)},

            {-0.5 * l2 * (zeta * zeta + zeta),
             0.5 * (l0 - l2) * (zeta * zeta + zeta),
             0.5 * l2 * l0 * (2 * zeta + 1)},

            // Face-center nodes on quad faces (3)
            {0.5 * (l1 - l0) * (1 - zeta * zeta),
             -0.5 * l1 * (1 - zeta * zeta),
             -l0 * l1 * zeta},

            {0.5 * l2 * (1 - zeta * zeta),
             0.5 * l1 * (1 - zeta * zeta),
             -l1 * l2 * zeta},

            {-0.5 * l2 * (1 - zeta * zeta),
             0.5 * (l0 - l2) * (1 - zeta * zeta),
             -l2 * l0 * zeta}
        };
    }

    static constexpr std::array<QuadraturePointAndWeight, 6> quadraturePoints()
    {
        constexpr Real a = 2. / 3.;
        constexpr Real b = 1. / 6.;
        // constexpr Real c = 1. / std::sqrt(3.);
        constexpr Real c { 0.57735026919 };
        constexpr Real w = 1. / 6.;

        constexpr sofa::type::Vec<TopologicalDimension, Real> q0(b, b, -c);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q1(a, b, -c);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q2(b, a, -c);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q3(b, b, c);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q4(a, b, c);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q5(b, a, c);

        constexpr std::array<QuadraturePointAndWeight, 6> q {
            std::make_pair(q0, w),
            std::make_pair(q1, w),
            std::make_pair(q2, w),
            std::make_pair(q3, w),
            std::make_pair(q4, w),
            std::make_pair(q5, w)
        };
        return q;
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_PRISM_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticPrism, sofa::defaulttype::Vec3Types>;
#endif

}
