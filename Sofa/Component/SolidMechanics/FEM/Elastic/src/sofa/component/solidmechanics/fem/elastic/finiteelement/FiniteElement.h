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
