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
#include <sofa/geometry/Pyramid.h>

#if !defined(SOFA_FEM_FINITE_ELEMENT_PYRAMID_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::Pyramid, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::Pyramid, DataTypes, 3);
    static_assert(spatial_dimensions == 3, "Pyramids are only defined in 3D");

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        {-1, -1, -1},
        { 1, -1, -1},
        { 1,  1, -1},
        {-1,  1, -1},
        { 0,  0,  1},
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getPyramids();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        return {
            static_cast<Real>(0.125) * (1 - q[0]) * (1 - q[1]) * (1 - q[2]),
            static_cast<Real>(0.125) * (1 + q[0]) * (1 - q[1]) * (1 - q[2]),
            static_cast<Real>(0.125) * (1 + q[0]) * (1 + q[1]) * (1 - q[2]),
            static_cast<Real>(0.125) * (1 - q[0]) * (1 + q[1]) * (1 - q[2]),
            static_cast<Real>(0.5)   * (1 + q[2])
        };
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        return {
            {-static_cast<Real>(0.125) * (1 - q[1]) * (1 - q[2]), -static_cast<Real>(0.125) * (1 - q[0]) * (1 - q[2]), -static_cast<Real>(0.125) * (1 - q[0]) * (1 - q[1])},
            { static_cast<Real>(0.125) * (1 - q[1]) * (1 - q[2]), -static_cast<Real>(0.125) * (1 + q[0]) * (1 - q[2]), -static_cast<Real>(0.125) * (1 + q[0]) * (1 - q[1])},
            { static_cast<Real>(0.125) * (1 + q[1]) * (1 - q[2]),  static_cast<Real>(0.125) * (1 + q[0]) * (1 - q[2]), -static_cast<Real>(0.125) * (1 + q[0]) * (1 + q[1])},
            {-static_cast<Real>(0.125) * (1 + q[1]) * (1 - q[2]),  static_cast<Real>(0.125) * (1 - q[0]) * (1 - q[2]), -static_cast<Real>(0.125) * (1 - q[0]) * (1 + q[1])},
            { 0,                                                 0,                                                 static_cast<Real>(0.5)}
        };
    }

    static constexpr auto quadraturePoints()
    {
        constexpr Real sqrt3_1 = static_cast<Real>(1) / static_cast<Real>(1.73205080757);
        constexpr Real one = static_cast<Real>(1);

        // We use the 8 Gauss points of the hexahedron, which exactly integrate the (1-z)^2 Jacobian of the pyramid.
        return std::array {
            std::pair{ReferenceCoord{-sqrt3_1, -sqrt3_1, -sqrt3_1}, one},
            std::pair{ReferenceCoord{ sqrt3_1, -sqrt3_1, -sqrt3_1}, one},
            std::pair{ReferenceCoord{ sqrt3_1,  sqrt3_1, -sqrt3_1}, one},
            std::pair{ReferenceCoord{-sqrt3_1,  sqrt3_1, -sqrt3_1}, one},
            std::pair{ReferenceCoord{-sqrt3_1, -sqrt3_1,  sqrt3_1}, one},
            std::pair{ReferenceCoord{ sqrt3_1, -sqrt3_1,  sqrt3_1}, one},
            std::pair{ReferenceCoord{ sqrt3_1,  sqrt3_1,  sqrt3_1}, one},
            std::pair{ReferenceCoord{-sqrt3_1,  sqrt3_1,  sqrt3_1}, one},
        };
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_PYRAMID_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::Pyramid, sofa::defaulttype::Vec3Types>;
#endif

}
