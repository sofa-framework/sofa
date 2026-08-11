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

#if !defined(SOFA_FEM_FINITE_ELEMENT_EDGE_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::Edge, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::Edge, DataTypes, 1);

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{ReferenceCoord{-1}, ReferenceCoord{1}}};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getEdges();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        return {
            static_cast<Real>(0.5) * (static_cast<Real>(1) - q[0]),
            static_cast<Real>(0.5) * (static_cast<Real>(1) + q[0])
        };
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        SOFA_UNUSED(q);
        return {{-static_cast<Real>(0.5)}, {static_cast<Real>(0.5)}};
    }

    template <sofa::Size Degree = 1>
    static constexpr auto quadraturePoints()
    {
        if constexpr (Degree <= 1)
        {
            // Degree 1: 1-point midpoint rule (default).
            return std::array<QuadraturePointAndWeight, 1>{
                std::make_pair(ReferenceCoord(static_cast<Real>(0)), static_cast<Real>(2))
            };
        }
        else if constexpr (Degree <= 3)
        {
            // Degrees 2-3: 2-point Gauss-Legendre rule.
            constexpr Real sqrt3 = 1.73205080757;
            constexpr Real g = static_cast<Real>(1) / sqrt3;
            return std::array<QuadraturePointAndWeight, 2>{
                std::make_pair(ReferenceCoord(-g), static_cast<Real>(1)),
                std::make_pair(ReferenceCoord( g), static_cast<Real>(1))
            };
        }
        else
        {
            static_assert(Degree <= 3, "FiniteElement<Edge>: no quadrature rule for the requested degree");
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
            default:
                throw std::invalid_argument("FiniteElement<Edge>::quadratureRule: unsupported degree");
        }
    }

};

#if !defined(SOFA_FEM_FINITE_ELEMENT_EDGE_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::Edge, sofa::defaulttype::Vec3Types>;
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::Edge, sofa::defaulttype::Vec2Types>;
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::Edge, sofa::defaulttype::Vec1Types>;
#endif

}
