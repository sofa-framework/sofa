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

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_EDGE_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::QuadraticEdge, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::QuadraticEdge, DataTypes, 1);

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        ReferenceCoord{0},    // vertex 0
        ReferenceCoord{1},    // vertex 1
        ReferenceCoord{0.5}   // mid-edge node
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getElements<sofa::geometry::QuadraticEdge>();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real xi = q[0];
        return {
            2 * xi * (xi - 0.5),      // vertex 0: (2*xi - 1) * xi
            2 * xi * (xi - 0.5) + 1 - 2 * xi,  // vertex 1: (2*xi - 1) * (xi - 1) = 2*xi^2 - 3*xi + 1
            4 * xi * (1 - xi)         // mid-edge: 4*xi*(1-xi)
        };
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real xi = q[0];
        return {
            {4 * xi - 3},    // vertex 0
            {4 * xi - 1},    // vertex 1
            {4 - 8 * xi}     // mid-edge
        };
    }

    static constexpr std::array<QuadraturePointAndWeight, 2> quadraturePoints()
    {
        // constexpr Real a = (1. - 1. / std::sqrt(3.)) / 2.;
        constexpr Real a { 0.211324865405 };
        // constexpr Real b = (1. + 1. / std::sqrt(3.)) / 2.;
        constexpr Real b { 0.788675134595 };
        constexpr Real w = 0.5;

        constexpr sofa::type::Vec<TopologicalDimension, Real> q0(a);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q1(b);

        constexpr std::array<QuadraturePointAndWeight, 2> q {
            std::make_pair(q0, w),
            std::make_pair(q1, w)
        };
        return q;
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_EDGE_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticEdge, sofa::defaulttype::Vec1Types>;
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticEdge, sofa::defaulttype::Vec2Types>;
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticEdge, sofa::defaulttype::Vec3Types>;
#endif

}
