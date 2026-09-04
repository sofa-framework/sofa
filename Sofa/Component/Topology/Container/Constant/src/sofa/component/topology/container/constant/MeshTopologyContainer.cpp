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
#include <sofa/component/topology/container/constant/MeshTopologyContainer.h>

namespace sofa::component::topology::container::constant
{

using sofa::geometry::ElementType;

constexpr const char* elementTypeToDataName(ElementType type)
{
    switch (type)
    {
        case ElementType::POINT: { return "position"; }
        case ElementType::EDGE: { return "edges"; }
        case ElementType::TRIANGLE: { return "triangles"; }
        case ElementType::QUAD: { return "quads"; }
        case ElementType::TETRAHEDRON: { return "tetrahedra"; }
        case ElementType::HEXAHEDRON: { return "hexahedra"; }
        case ElementType::PRISM: { return "prisms"; }
        case ElementType::PYRAMID: { return "pyramids"; }

        case ElementType::QUADRATIC_EDGE: { return "quadratic_edges"; }
        case ElementType::QUADRATIC_TRIANGLE: { return "quadratic_triangles"; }
        case ElementType::QUADRATIC_QUAD: { return "quadratic_quads"; }
        case ElementType::QUADRATIC_TETRAHEDRON: { return "quadratic_tetrahedra"; }
        case ElementType::QUADRATIC_HEXAHEDRON: { return "quadratic_hexahedra"; }
        case ElementType::QUADRATIC_PRISM: { return "quadratic_prisms"; }
        case ElementType::QUADRATIC_PYRAMID: { return "quadratic_pyramids"; }

        default:
            return "Unknown";
    }
}
template<std::size_t I>
void registerSingleData(core::objectmodel::BaseComponent* meshTopology, auto& dataContainer)
{
    using AllElements = std::remove_cvref_t<decltype(dataContainer)>;
    using DataSeqElementPtr = std::tuple_element_t<I, AllElements>;
    using DataSeqElement = typename DataSeqElementPtr::element_type;
    using SeqElement = typename DataSeqElement::value_type;
    using TopologyElement = typename SeqElement::value_type;
    static constexpr auto elementType = TopologyElement::Element_type;

    const char* elementName = sofa::geometry::elementTypeToString(elementType);
    auto elementNameStr = sofa::helper::downcaseString(std::string(elementName));

    auto dataName = elementTypeToDataName(elementType);
    std::string dataNameStr { dataName };
    sofa::helper::replaceAll(dataNameStr, " ", "_");

    auto& dataPtr = std::get<I>(dataContainer);
    dataPtr = std::make_unique<DataSeqElement>("List of " + elementNameStr);

    sofa::core::objectmodel::BaseData& data = *dataPtr;
    data.setName(dataName);

    meshTopology->addData(dataPtr.get(), dataName);
}

template<size_t... Is>
void registerData(core::objectmodel::BaseComponent* meshTopology, auto& dataContainer, std::index_sequence<Is...>)
{
    (registerSingleData<Is>(meshTopology, dataContainer), ...);
}

const void* MeshTopologyContainer::getElementsRaw(
    const sofa::geometry::ElementType& elementType) const noexcept
{
    switch (elementType)
    {
        case ElementType::UNKNOWN:
        case ElementType::POINT:
            break;
        case ElementType::EDGE:
            return &get<sofa::geometry::Edge>()->getValue();
        case ElementType::TRIANGLE:
            return &get<sofa::geometry::Triangle>()->getValue();
        case ElementType::QUAD:
            return &get<sofa::geometry::Quad>()->getValue();
        case ElementType::TETRAHEDRON:
            return &get<sofa::geometry::Tetrahedron>()->getValue();
        case ElementType::HEXAHEDRON:
            return &get<sofa::geometry::Hexahedron>()->getValue();
        case ElementType::PRISM:
            return &get<sofa::geometry::Prism>()->getValue();
        case ElementType::PYRAMID:
            return &get<sofa::geometry::Pyramid>()->getValue();
        case ElementType::QUADRATIC_EDGE:
            return &get<sofa::geometry::QuadraticEdge>()->getValue();
        case ElementType::QUADRATIC_TRIANGLE:
            return &get<sofa::geometry::QuadraticTriangle>()->getValue();
        case ElementType::QUADRATIC_QUAD:
            return &get<sofa::geometry::QuadraticQuad>()->getValue();
        case ElementType::QUADRATIC_TETRAHEDRON:
            return &get<sofa::geometry::QuadraticTetrahedron>()->getValue();
        case ElementType::QUADRATIC_HEXAHEDRON:
            return &get<sofa::geometry::QuadraticHexahedron>()->getValue();
        case ElementType::QUADRATIC_PRISM:
            return &get<sofa::geometry::QuadraticPrism>()->getValue();
        case ElementType::QUADRATIC_PYRAMID:
            return &get<sofa::geometry::QuadraticPyramid>()->getValue();
        case ElementType::SIZE:
        default:
            return nullptr;
    }
    return nullptr;
}

MeshTopologyContainer::MeshTopologyContainer() : Inherit1()
{
    registerData(this, m_container, std::make_index_sequence<std::tuple_size_v<AllElements>>{});
}

const core::topology::BaseMeshTopology::SeqEdges& MeshTopologyContainer::getEdges()
{
    return get<sofa::geometry::Edge>()->getValue();
}
const core::topology::BaseMeshTopology::SeqTriangles& MeshTopologyContainer::getTriangles()
{
    return get<sofa::geometry::Triangle>()->getValue();
}
const core::topology::BaseMeshTopology::SeqQuads& MeshTopologyContainer::getQuads()
{
    return get<sofa::geometry::Quad>()->getValue();
}
const core::topology::BaseMeshTopology::SeqTetrahedra& MeshTopologyContainer::getTetrahedra()
{
    return get<sofa::geometry::Tetrahedron>()->getValue();
}
const core::topology::BaseMeshTopology::SeqHexahedra& MeshTopologyContainer::getHexahedra()
{
    return get<sofa::geometry::Hexahedron>()->getValue();
}
const core::topology::BaseMeshTopology::SeqPrisms& MeshTopologyContainer::getPrisms()
{
    return get<sofa::geometry::Prism>()->getValue();
}
const core::topology::BaseMeshTopology::SeqPyramids& MeshTopologyContainer::getPyramids()
{
    return get<sofa::geometry::Pyramid>()->getValue();
}

}  // namespace sofa::component::topology::container::constant
