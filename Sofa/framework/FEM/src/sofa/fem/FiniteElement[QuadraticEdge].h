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
    FINITEELEMENT_HEADER(sofa::geometry::QuadraticEdge, DataTypes, 1, 3);

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        ReferenceCoord{-1},   // vertex 0
        ReferenceCoord{1},    // vertex 1
        ReferenceCoord{0}     // mid-edge node
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getElements<sofa::geometry::QuadraticEdge>();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real xi = q[0];
        constexpr Real half = static_cast<Real>(0.5);
        return {
            half * xi * (xi - 1),                  // vertex 0, at xi = -1
            half * xi * (xi + 1),                  // vertex 1, at xi = 1
            static_cast<Real>(1) - xi * xi         // mid-edge node, at xi = 0
        };
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const Real xi = q[0];
        constexpr Real half = static_cast<Real>(0.5);
        return {
            {xi - half},                   // vertex 0
            {xi + half},                   // vertex 1
            {-2 * xi}                      // mid-edge node
        };
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_EDGE_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticEdge, sofa::defaulttype::Vec1Types>;
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticEdge, sofa::defaulttype::Vec2Types>;
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticEdge, sofa::defaulttype::Vec3Types>;
#endif

}
