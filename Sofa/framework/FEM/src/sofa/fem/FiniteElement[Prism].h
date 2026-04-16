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

#if !defined(SOFA_FEM_FINITE_ELEMENT_PRISM_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::Prism, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::Prism, DataTypes, 3);
    static_assert(spatial_dimensions == 3, "Prisms are only defined in 3D");

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 1},
        {0, 1, 1},
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getPrisms();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        return {
            q[0] * q[2] - q[0] + q[1] * q[2] - q[1] - q[2] + 1,
            (1 - q[2]) * q[0],
            (1 - q[2]) * q[1],
            (-q[0] - q[1] + 1) * q[2],
            q[0] * q[2],
            q[1] * q[2]
        };
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        SOFA_UNUSED(q);
        return {
            {q[2] - 1, q[2] - 1, q[0] + q[1] - 1},
            {1 - q[2], 0, -q[0]},
            {0, 1 - q[2], -q[1]},
            {-q[2], -q[2], -q[0] - q[1] + 1},
            {q[2], 0, q[0]},
            {0, q[2], q[1]},
        };
    }

    static constexpr std::array<QuadraturePointAndWeight, 2> quadraturePoints()
    {
        constexpr auto third = static_cast<Real>(1) / static_cast<Real>(3);
        constexpr auto sqrt_3 = static_cast<Real>(0.57735026919); // 1/sqrt(3)
        constexpr auto one = static_cast<Real>(1);
        constexpr QuadraturePoint q0 {third, third, static_cast<Real>(0.5) * (one - sqrt_3)};
        constexpr QuadraturePoint q1 {third, third, static_cast<Real>(0.5) * (one + sqrt_3)};

        constexpr std::array<QuadraturePointAndWeight, 2> q {
            std::make_pair(q0, 1./4.),
            std::make_pair(q1, 1./4.),
        };
        return q;
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_PRISM_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::Prism, sofa::defaulttype::Vec3Types>;
#endif

}
