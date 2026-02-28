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
#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement.h>

namespace sofa::component::solidmechanics::fem::elastic
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

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        return {
            {1 / static_cast<Real>(4) * (-static_cast<Real>(1) + q[1]), 1 / static_cast<Real>(4) * (-static_cast<Real>(1) + q[0])},
            {1 / static_cast<Real>(4) * ( static_cast<Real>(1) - q[1]), 1 / static_cast<Real>(4) * (-static_cast<Real>(1) - q[0])},
            {1 / static_cast<Real>(4) * ( static_cast<Real>(1) + q[1]), 1 / static_cast<Real>(4) * ( static_cast<Real>(1) + q[0])},
            {1 / static_cast<Real>(4) * (-static_cast<Real>(1) - q[1]), 1 / static_cast<Real>(4) * ( static_cast<Real>(1) - q[0])}
        };
    }

    static constexpr std::array<QuadraturePointAndWeight, 3> quadraturePoints()
    {
        constexpr Real sqrt2_3 = 0.816496580928; //sqrt(2./3.)
        constexpr Real sqrt6 = 2.44948974278; //sqrt(6.)
        constexpr Real sqrt2 = 1.41421356237; //sqrt(2.)

        constexpr sofa::type::Vec<TopologicalDimension, Real> q0(sqrt2_3, 0.);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q1(-1/sqrt6, -1./sqrt2);
        constexpr sofa::type::Vec<TopologicalDimension, Real> q2(-1/sqrt6, 1./sqrt2);

        return {
            std::make_pair(q0, 4./3.),
            std::make_pair(q1, 4./3.),
            std::make_pair(q2, 4./3.),
        };
    }
};

}
