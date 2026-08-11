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

#if !defined(SOFA_FEM_FINITE_ELEMENT_TETAHEDRON_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::Tetrahedron, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::Tetrahedron, DataTypes, 3);
    static_assert(spatial_dimensions == 3, "Tetrahedrons are only defined in 3D");

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getTetrahedra();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        return {
            static_cast<Real>(1) - q[0] - q[1] - q[2],
            q[0],
            q[1],
            q[2]
        };
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        SOFA_UNUSED(q);
        return {
            {-1, -1, -1},
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        };
    }

    template <sofa::Size Degree = 1>
    static constexpr auto quadraturePoints()
    {
        if constexpr (Degree <= 1)
        {
            // Degree 1: 1-point centroid rule (default).
            return std::array<QuadraturePointAndWeight, 1>{
                std::make_pair(ReferenceCoord(1./4., 1./4., 1./4.), Real(1./6.))
            };
        }
        else if constexpr (Degree <= 2)
        {
            // Degree 2: 4-point rule.
            constexpr Real sqrt5 = 2.2360679774997896;
            constexpr Real a = (5. - sqrt5) / 20.;
            constexpr Real b = (5. + 3. * sqrt5) / 20.;
            return std::array<QuadraturePointAndWeight, 4>{
                std::make_pair(ReferenceCoord(a, a, a), Real(1./24.)),
                std::make_pair(ReferenceCoord(b, a, a), Real(1./24.)),
                std::make_pair(ReferenceCoord(a, b, a), Real(1./24.)),
                std::make_pair(ReferenceCoord(a, a, b), Real(1./24.))
            };
        }
        else
        {
            static_assert(Degree <= 2, "FiniteElement<Tetrahedron>: no quadrature rule for the requested degree");
        }
    }

    // Quadrature rule selector by degree; view of the compile-time table.
    static std::span<const QuadraturePointAndWeight> quadratureRule(sofa::Size degree)
    {
        switch (degree)
        {
            case 1: { static constexpr auto rule = quadraturePoints<1>(); return rule; }
            case 2: { static constexpr auto rule = quadraturePoints<2>(); return rule; }
            default:
                throw std::invalid_argument("FiniteElement<Tetrahedron>::quadratureRule: unsupported degree");
        }
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_TETAHEDRON_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::Tetrahedron, sofa::defaulttype::Vec3Types>;
#endif

}
