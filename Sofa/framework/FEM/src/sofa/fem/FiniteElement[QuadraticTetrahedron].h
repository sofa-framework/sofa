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

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_TETAHEDRON_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::QuadraticTetrahedron, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::QuadraticTetrahedron, DataTypes, 3);
    static_assert(spatial_dimensions == 3, "Quadratic Tetrahedrons are only defined in 3D");

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        {0, 0, 0},  // vertex 0
        {1, 0, 0},  // vertex 1
        {0, 1, 0},  // vertex 2
        {0, 0, 1},  // vertex 3
        {0.5, 0, 0},  // mid-edge 0-1
        {0, 0.5, 0},  // mid-edge 0-2
        {0, 0, 0.5},  // mid-edge 0-3
        {0.5, 0.5, 0},  // mid-edge 1-2
        {0.5, 0, 0.5},  // mid-edge 1-3
        {0, 0.5, 0.5}   // mid-edge 2-3
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getElements<sofa::geometry::QuadraticTetrahedron>();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real l0 = static_cast<Real>(1) - q[0] - q[1] - q[2];
        const Real l1 = q[0];
        const Real l2 = q[1];
        const Real l3 = q[2];

        return {
            l0 * (2 * l0 - 1),  // vertex 0
            l1 * (2 * l1 - 1),  // vertex 1
            l2 * (2 * l2 - 1),  // vertex 2
            l3 * (2 * l3 - 1),  // vertex 3
            4 * l0 * l1,        // mid-edge 0-1
            4 * l0 * l2,        // mid-edge 0-2
            4 * l0 * l3,        // mid-edge 0-3
            4 * l1 * l2,        // mid-edge 1-2
            4 * l1 * l3,        // mid-edge 1-3
            4 * l2 * l3         // mid-edge 2-3
        };
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real l0 = static_cast<Real>(1) - q[0] - q[1] - q[2];
        const Real l1 = q[0];
        const Real l2 = q[1];
        const Real l3 = q[2];

        return {
            // vertex 0: derivatives with respect to q[0], q[1], q[2]
            {-3 + 4 * (q[0] + q[1] + q[2]), -3 + 4 * (q[0] + q[1] + q[2]), -3 + 4 * (q[0] + q[1] + q[2])},
            // vertex 1
            {4 * q[0] - 1, 0, 0},
            // vertex 2
            {0, 4 * q[1] - 1, 0},
            // vertex 3
            {0, 0, 4 * q[2] - 1},
            // mid-edge 0-1
            {4 * (l0 - l1), -4 * l1, -4 * l1},
            // mid-edge 0-2
            {-4 * l2, 4 * (l0 - l2), -4 * l2},
            // mid-edge 0-3
            {-4 * l3, -4 * l3, 4 * (l0 - l3)},
            // mid-edge 1-2
            {4 * l2, 4 * l1, 0},
            // mid-edge 1-3
            {4 * l3, 0, 4 * l1},
            // mid-edge 2-3
            {0, 4 * l3, 4 * l2}
        };
    }

    static constexpr std::array<QuadraturePointAndWeight, 4> quadraturePoints()
    {
        // 4-point quadrature rule for quadratic tetrahedron
        // constexpr Real a = (5. + 3. * std::sqrt(5.)) / 20.;
        constexpr Real a { 0.585410196625 };
        // constexpr Real b = (5. - std::sqrt(5.)) / 20.;
        constexpr Real b { 0.138196601125 };
        constexpr Real w = 1. / 24.;

        constexpr sofa::type::Vec<TopologicalDimension, Real> q0(a, b, b);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q1(b, a, b);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q2(b, b, a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q3(b, b, b);

        constexpr std::array<QuadraturePointAndWeight, 4> q {
            std::make_pair(q0, w),
            std::make_pair(q1, w),
            std::make_pair(q2, w),
            std::make_pair(q3, w)
        };
        return q;
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_TETAHEDRON_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticTetrahedron, sofa::defaulttype::Vec3Types>;
#endif

}
