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

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUAD_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::Quad, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::Quad, DataTypes, 2);
    static_assert(spatial_dimensions > 1, "Quads cannot be defined in 1D");

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        {-1, -1},
        {1, -1},
        {1, 1},
        {-1, 1}
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getQuads();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        return {
            static_cast<Real>(0.25) * (q[0] - 1) * (q[1] - 1),
            -static_cast<Real>(0.25) * (q[0] + 1) * (q[1] - 1),
            static_cast<Real>(0.25) * (q[0] + 1) * (q[1] + 1),
            -static_cast<Real>(0.25) * (q[0] - 1) * (q[1] + 1)
        };
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        return {
            {1 / static_cast<Real>(4) * (-static_cast<Real>(1) + q[1]), 1 / static_cast<Real>(4) * (-static_cast<Real>(1) + q[0])},
            {1 / static_cast<Real>(4) * ( static_cast<Real>(1) - q[1]), 1 / static_cast<Real>(4) * (-static_cast<Real>(1) - q[0])},
            {1 / static_cast<Real>(4) * ( static_cast<Real>(1) + q[1]), 1 / static_cast<Real>(4) * ( static_cast<Real>(1) + q[0])},
            {1 / static_cast<Real>(4) * (-static_cast<Real>(1) - q[1]), 1 / static_cast<Real>(4) * ( static_cast<Real>(1) - q[0])}
        };
    }

    template <sofa::Size Degree = 2>
    static constexpr auto quadraturePoints()
    {
        if constexpr (Degree <= 1)
        {
            // Degree 1: 1-point centroid rule.
            return std::array<QuadraturePointAndWeight, 1>{
                std::make_pair(ReferenceCoord(static_cast<Real>(0), static_cast<Real>(0)), static_cast<Real>(4))
            };
        }
        else if constexpr (Degree <= 2)
        {
            // Degree 2: 3-point rule (default).
            constexpr Real sqrt2_3 = 0.816496580928; //sqrt(2./3.)
            constexpr Real sqrt6 = 2.44948974278; //sqrt(6.)
            constexpr Real sqrt2 = 1.41421356237; //sqrt(2.)

            constexpr ReferenceCoord q0(sqrt2_3, 0.);
            constexpr ReferenceCoord q1(-1/sqrt6, -1./sqrt2);
            constexpr ReferenceCoord q2(-1/sqrt6, 1./sqrt2);

            return std::array<QuadraturePointAndWeight, 3>{
                std::make_pair(q0, 4./3.),
                std::make_pair(q1, 4./3.),
                std::make_pair(q2, 4./3.)
            };
        }
        else if constexpr (Degree <= 3)
        {
            // Degree 3: 2x2 Gauss-Legendre rule.
            constexpr Real sqrt3 = 1.73205080757; //sqrt(3.)
            constexpr Real g = static_cast<Real>(1) / sqrt3;
            return std::array<QuadraturePointAndWeight, 4>{
                std::make_pair(ReferenceCoord(-g, -g), static_cast<Real>(1)),
                std::make_pair(ReferenceCoord( g, -g), static_cast<Real>(1)),
                std::make_pair(ReferenceCoord( g,  g), static_cast<Real>(1)),
                std::make_pair(ReferenceCoord(-g,  g), static_cast<Real>(1))
            };
        }
        else
        {
            static_assert(Degree <= 3, "FiniteElement<Quad>: no quadrature rule for the requested degree");
        }
    }

    // Quadrature rule selector by degree; view of the compile-time table.
    static std::span<const QuadraturePointAndWeight> quadratureRule(sofa::Size degree)
    {
        switch (degree)
        {
            case 1: { static constexpr auto rule = quadraturePoints<1>(); return rule; }
            case 2: { static constexpr auto rule = quadraturePoints<2>(); return rule; }
            case 3: { static constexpr auto rule = quadraturePoints<3>(); return rule; }
            default:
                throw std::invalid_argument("FiniteElement<Quad>::quadratureRule: unsupported degree");
        }
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUAD_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::Quad, sofa::defaulttype::Vec3Types>;
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::Quad, sofa::defaulttype::Vec2Types>;
#endif

}
