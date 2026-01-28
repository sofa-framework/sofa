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
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class ElementType, class DataTypes>
struct FiniteElement;

#define FINITEELEMENT_HEADER(ElType, DataTypes, dimension) \
    using Coord = sofa::Coord_t<DataTypes>;\
    using Real = sofa::Real_t<DataTypes>;\
    using ElementType = ElType;\
    using TopologyElement = sofa::topology::Element<ElementType>;\
    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;\
    static constexpr sofa::Size NumberOfNodesInElement = ElementType::NumberOfNodes;\
    static constexpr sofa::Size TopologicalDimension = dimension;\
    using ReferenceCoord = sofa::type::Vec<TopologicalDimension, Real>;\
    using ShapeFunctionType = std::function<Real(const ReferenceCoord&)>;\
    using QuadraturePoint = ReferenceCoord; \
    using QuadraturePointAndWeight = std::pair<QuadraturePoint, Real>


template <class ElementType, class DataTypes>
struct FiniteElementHelper
{
    using FiniteElement = sofa::component::solidmechanics::fem::elastic::FiniteElement<ElementType, DataTypes>;
    using Coord = typename FiniteElement::Coord;
    using Real = typename FiniteElement::Real;

    static constexpr sofa::Size spatial_dimensions = FiniteElement::spatial_dimensions;
    static constexpr sofa::Size NumberOfNodesInElement = FiniteElement::NumberOfNodesInElement;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;

    // gradient of shape functions in the reference element evaluated at the quadrature point
    static constexpr auto gradientShapeFunctionAtQuadraturePoints()
    {
        using Gradient = sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real>;
        constexpr auto quadraturePoints = FiniteElement::quadraturePoints();

        std::array<Gradient, quadraturePoints.size()> gradients;
        std::transform(quadraturePoints.begin(), quadraturePoints.end(), gradients.begin(),
            [](const auto& qp) { return FiniteElement::gradientShapeFunctions(qp.first); });
        return gradients;
    }

    // jacobian of the mapping from the reference space to the physical space, evaluated where the
    // gradient of the shape functions has been evaluated.
    static constexpr auto jacobianFromReferenceToPhysical(
        const std::array<Coord, NumberOfNodesInElement>& elementNodesCoordinates,
        const sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real>& gradientShapeFunctionInReferenceElement)
    {
        sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real> jacobian;
        for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
        {
            jacobian += sofa::type::dyad(elementNodesCoordinates[i], gradientShapeFunctionInReferenceElement[i]);
        }
        return jacobian;
    }

};

}
