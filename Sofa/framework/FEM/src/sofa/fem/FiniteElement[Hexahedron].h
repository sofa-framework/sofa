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
#include <span>
#include <stdexcept>

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

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        return {
            -static_cast<Real>(0.125) * (q[0] - 1) * (q[1] - 1) * (q[2] - 1),
             static_cast<Real>(0.125) * (q[0] + 1) * (q[1] - 1) * (q[2] - 1),
            -static_cast<Real>(0.125) * (q[0] + 1) * (q[1] + 1) * (q[2] - 1),
             static_cast<Real>(0.125) * (q[0] - 1) * (q[1] + 1) * (q[2] - 1),
             static_cast<Real>(0.125) * (q[0] - 1) * (q[1] - 1) * (q[2] + 1),
            -static_cast<Real>(0.125) * (q[0] + 1) * (q[1] - 1) * (q[2] + 1),
             static_cast<Real>(0.125) * (q[0] + 1) * (q[1] + 1) * (q[2] + 1),
            -static_cast<Real>(0.125) * (q[0] - 1) * (q[1] + 1) * (q[2] + 1),
        };
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

    template <sofa::Size Degree = 3>
    static constexpr auto quadraturePoints()
    {
        if constexpr (Degree <= 1)
        {
            // Degree 1: 1-point centroid rule.
            return std::array<QuadraturePointAndWeight, 1>{
                std::make_pair(ReferenceCoord(static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(0)), static_cast<Real>(8))
            };
        }
        else if constexpr (Degree <= 3)
        {
            // Degrees 2-3: 2x2x2 Gauss-Legendre rule (default).
            constexpr Real sqrt3 = 1.73205080757; //sqrt(3.)
            constexpr Real sqrt3_1 = static_cast<Real>(1) / sqrt3;
            constexpr Real one = static_cast<Real>(1);

            return std::array {
                std::pair{referenceElementNodes[0] * sqrt3_1, one},
                std::pair{referenceElementNodes[1] * sqrt3_1, one},
                std::pair{referenceElementNodes[2] * sqrt3_1, one},
                std::pair{referenceElementNodes[3] * sqrt3_1, one},
                std::pair{referenceElementNodes[4] * sqrt3_1, one},
                std::pair{referenceElementNodes[5] * sqrt3_1, one},
                std::pair{referenceElementNodes[6] * sqrt3_1, one},
                std::pair{referenceElementNodes[7] * sqrt3_1, one},
            };
        }
        else if constexpr (Degree <= 5)
        {
            // Degrees 4-5: 3x3x3 Gauss-Legendre rule.
            constexpr Real g = 0.77459666924; //sqrt(3./5.)
            constexpr std::array<Real, 3> node{ -g, static_cast<Real>(0), g };
            constexpr std::array<Real, 3> weight{ static_cast<Real>(5./9.), static_cast<Real>(8./9.), static_cast<Real>(5./9.) };

            std::array<QuadraturePointAndWeight, 27> q{};
            sofa::Size k = 0;
            for (sofa::Size i = 0; i < 3; ++i)
                for (sofa::Size j = 0; j < 3; ++j)
                    for (sofa::Size l = 0; l < 3; ++l)
                        q[k++] = std::make_pair(ReferenceCoord(node[i], node[j], node[l]), weight[i] * weight[j] * weight[l]);
            return q;
        }
        else
        {
            static_assert(Degree <= 5, "FiniteElement<Hexahedron>: no quadrature rule for the requested degree");
        }
    }

    // Quadrature rule selector by degree; view of the compile-time table.
    static std::span<const QuadraturePointAndWeight> quadratureRule(sofa::Size degree)
    {
        switch (degree)
        {
            case 1: { static constexpr auto rule = quadraturePoints<1>(); return rule; }
            case 2:
            case 3: { static constexpr auto rule = quadraturePoints<3>(); return rule; }
            case 4:
            case 5: { static constexpr auto rule = quadraturePoints<5>(); return rule; }
            default:
                throw std::invalid_argument("FiniteElement<Hexahedron>::quadratureRule: unsupported degree");
        }
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_HEXAHEDRON_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::Hexahedron, sofa::defaulttype::Vec3Types>;
#endif

}
