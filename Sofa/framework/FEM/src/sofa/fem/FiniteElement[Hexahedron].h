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

#if !defined(SOFA_FEM_FINITE_ELEMENT_HEXAHEDRON_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::Hexahedron, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::Hexahedron, DataTypes, 3);
    static_assert(spatial_dimensions == 3, "Hexahedrons are only defined in 3D");

    // Following the convention in sofa::geometry::Hexahedron:
    //     Y  n3---------n2
    //     ^  /          /|
    //     | /          / |
    //     n7---------n6  |
    //     |          |   |
    //     |  n0------|--n1
    //     | /        | /
    //     |/         |/
    //     n4---------n5-->X
    //    /
    //   /
    //  Z
    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        {-1, -1, -1},
        {1, -1, -1},
        {1, 1, -1},
        {-1, 1, -1},
        {-1, -1, 1},
        {1, -1, 1},
        {1, 1, 1},
        {-1, 1, 1},
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getHexahedra();
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const auto [x, y, z] = q;
        sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradient(sofa::type::NOINIT);
        using Line = typename sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real>::Line;

        for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
        {
            const auto& [xref, yref, zref] = referenceElementNodes[i];
            gradient[i] = 1./8. * Line(
                xref * (1 + y * yref) * (1 + z * zref),
                yref * (1 + x * xref) * (1 + z * zref),
                zref * (1 + x * xref) * (1 + y * yref));
        }

        return gradient;
    }

    static constexpr auto quadraturePoints()
    {
        constexpr Real sqrt3 = 1.73205080757; //sqrt(3.)
        constexpr Real sqrt3_1 = static_cast<Real>(1) / sqrt3;
        constexpr Real one = static_cast<Real>(1);

        constexpr std::array q {
            std::pair{referenceElementNodes[0] * sqrt3_1, one},
            std::pair{referenceElementNodes[1] * sqrt3_1, one},
            std::pair{referenceElementNodes[2] * sqrt3_1, one},
            std::pair{referenceElementNodes[3] * sqrt3_1, one},
            std::pair{referenceElementNodes[4] * sqrt3_1, one},
            std::pair{referenceElementNodes[5] * sqrt3_1, one},
            std::pair{referenceElementNodes[6] * sqrt3_1, one},
            std::pair{referenceElementNodes[7] * sqrt3_1, one},
        };

        return q;
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_HEXAHEDRON_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::Hexahedron, sofa::defaulttype::Vec3Types>;
#endif

}
