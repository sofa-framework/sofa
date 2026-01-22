#pragma once
#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement.h>

namespace sofa::component::solidmechanics::fem::elastic
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

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        SOFA_UNUSED(q);
        return {{-static_cast<Real>(0.5)}, {static_cast<Real>(0.5)}};
    }

    static constexpr std::array<QuadraturePointAndWeight, 1> quadraturePoints()
    {
        constexpr sofa::type::Vec<TopologicalDimension, Real> q0(static_cast<Real>(0));
        return {
            std::make_pair(q0, static_cast<Real>(2))
        };
    }

};

}
