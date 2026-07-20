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

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_QUAD_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::QuadraticQuad, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::QuadraticQuad, DataTypes, 2);

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        {-1, -1},  // vertex 0
        {1, -1},   // vertex 1
        {1, 1},    // vertex 2
        {-1, 1},   // vertex 3
        {0, -1},   // mid-edge 0-1
        {1, 0},    // mid-edge 1-2
        {0, 1},    // mid-edge 2-3
        {-1, 0},   // mid-edge 3-0
        {0, 0}     // face center
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getElements<sofa::geometry::QuadraticQuad>();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real xi = q[0];
        const Real eta = q[1];

        return {
            0.25 * (xi * xi - xi) * (eta * eta - eta),     // vertex 0
            0.25 * (xi * xi + xi) * (eta * eta - eta),     // vertex 1
            0.25 * (xi * xi + xi) * (eta * eta + eta),     // vertex 2
            0.25 * (xi * xi - xi) * (eta * eta + eta),     // vertex 3
            0.5 * (1 - xi * xi) * (eta * eta - eta),       // mid-edge 0-1
            0.5 * (xi * xi + xi) * (1 - eta * eta),        // mid-edge 1-2
            0.5 * (1 - xi * xi) * (eta * eta + eta),       // mid-edge 2-3
            0.5 * (xi * xi - xi) * (1 - eta * eta),        // mid-edge 3-0
            (1 - xi * xi) * (1 - eta * eta)                // face center
        };
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real xi = q[0];
        const Real eta = q[1];

        return {
            // vertex 0
            {0.25 * (2 * xi - 1) * (eta * eta - eta), 0.25 * (xi * xi - xi) * (2 * eta - 1)},
            // vertex 1
            {0.25 * (2 * xi + 1) * (eta * eta - eta), 0.25 * (xi * xi + xi) * (2 * eta - 1)},
            // vertex 2
            {0.25 * (2 * xi + 1) * (eta * eta + eta), 0.25 * (xi * xi + xi) * (2 * eta + 1)},
            // vertex 3
            {0.25 * (2 * xi - 1) * (eta * eta + eta), 0.25 * (xi * xi - xi) * (2 * eta + 1)},
            // mid-edge 0-1
            {-xi * (eta * eta - eta), 0.5 * (1 - xi * xi) * (2 * eta - 1)},
            // mid-edge 1-2
            {0.5 * (2 * xi + 1) * (1 - eta * eta), -(xi * xi + xi) * eta},
            // mid-edge 2-3
            {-xi * (eta * eta + eta), 0.5 * (1 - xi * xi) * (2 * eta + 1)},
            // mid-edge 3-0
            {0.5 * (2 * xi - 1) * (1 - eta * eta), -(xi * xi - xi) * eta},
            // face center
            {-2 * xi * (1 - eta * eta), -2 * eta * (1 - xi * xi)}
        };
    }

    static constexpr std::array<QuadraturePointAndWeight, 4> quadraturePoints()
    {
        // constexpr Real a = 1. / std::sqrt(3.);
        constexpr Real a { 0.57735026919 };
        constexpr Real w = 1.0;

        constexpr sofa::type::Vec<TopologicalDimension, Real> q0(-a, -a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q1(a, -a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q2(a, a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q3(-a, a);

        constexpr std::array<QuadraturePointAndWeight, 4> q {
            std::make_pair(q0, w),
            std::make_pair(q1, w),
            std::make_pair(q2, w),
            std::make_pair(q3, w)
        };
        return q;
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_QUAD_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticQuad, sofa::defaulttype::Vec2Types>;
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticQuad, sofa::defaulttype::Vec3Types>;
#endif

}
