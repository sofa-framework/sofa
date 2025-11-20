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
#include <sofa/component/visual/VisualMesh.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::visual
{

void registerVisualMesh(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Render a mesh")
        .add<VisualMesh>()
    );
}

VisualMesh::VisualMesh()
    : d_position(initData(&d_position, "position", "The position of the vertices of mesh"))
    , d_elementSpace(initData(&d_elementSpace, 0.15_sreal, "elementSpace",
                              "The space between element (scalar between 0 and 1)"))
    , l_topology(initLink("topology", "Link to a topology containing elements"))
{
}


void VisualMesh::init()
{
    VisualModel::init();

    if (!this->isComponentStateInvalid())
    {
        this->validateTopology();
    }

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

void VisualMesh::drawTriangles(helper::visual::DrawTool* drawTool)
{
    if (!l_topology)
        return;

    const auto& elements = l_topology->getTriangles();

    m_renderedPoints.resize(elements.size() * sofa::geometry::Triangle::NumberOfNodes);

    const auto elementSpace = d_elementSpace.getValue();
    const auto& positionAccessor = sofa::helper::getReadAccessor(d_position);

    auto pointsIt = m_renderedPoints.begin();

    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];

        const sofa::type::Vec3 center = elementCenter(positionAccessor.ref(), element);

        std::array facetsInElement{i};
        setPoints(facetsInElement, elements, positionAccessor.ref(), center, elementSpace, pointsIt);
    }

    while (m_renderedColors.size() < elements.size() * sofa::geometry::Triangle::NumberOfNodes)
    {
        static constexpr std::array colors{
            type::makeHomogeneousArray<sofa::type::RGBAColor, sofa::geometry::Triangle::NumberOfNodes>( sofa::type::RGBAColor::green()),
            type::makeHomogeneousArray<sofa::type::RGBAColor, sofa::geometry::Triangle::NumberOfNodes>(sofa::type::RGBAColor::teal()),
            type::makeHomogeneousArray<sofa::type::RGBAColor, sofa::geometry::Triangle::NumberOfNodes>(sofa::type::RGBAColor::blue()),
        };

        const auto elementId = m_renderedColors.size() / sofa::geometry::Triangle::NumberOfNodes;
        const auto triangleColor = colors[elementId % colors.size()];

        m_renderedColors.insert(m_renderedColors.end(), triangleColor.begin(), triangleColor.end());
    }

    drawTool->drawTriangles(m_renderedPoints, m_renderedColors);
}

namespace
{
template<std::size_t NumberFacetsInElement, std::size_t NumberVerticesInFacet>
std::array<sofa::type::RGBAColor, NumberFacetsInElement * NumberVerticesInFacet>
constexpr generateElementColors(const std::array<sofa::type::RGBAColor, NumberFacetsInElement>& facetColors)
{
    std::array<sofa::type::RGBAColor, NumberFacetsInElement * NumberVerticesInFacet> verticeColors;
    auto verticeColorsIt = verticeColors.begin();
    for (const auto& c : facetColors)
    {
        for (std::size_t i = 0; i < NumberVerticesInFacet; ++i)
        {
            *verticeColorsIt++ = c;
        }
    }
    return verticeColors;
}
}

void VisualMesh::drawTetrahedra(helper::visual::DrawTool* drawTool)
{
    if (!l_topology)
        return;

    const auto& elements = l_topology->getTetrahedra();
    const auto& facets = l_topology->getTriangles();

    static constexpr std::size_t NumberTrianglesInTetrahedron = 4;
    m_renderedPoints.resize(elements.size() * NumberTrianglesInTetrahedron * sofa::geometry::Triangle::NumberOfNodes);

    const auto elementSpace = d_elementSpace.getValue();
    const auto& positionAccessor = sofa::helper::getReadAccessor(d_position);

    auto pointsIt = m_renderedPoints.begin();

    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];
        const auto& facetsInElement = l_topology->getTrianglesInTetrahedron(i);
        assert(facetsInElement.size() == NumberTrianglesInTetrahedron);

        const sofa::type::Vec3 center = elementCenter(positionAccessor.ref(), element);

        setPoints(facetsInElement, facets, positionAccessor.ref(), center, elementSpace, pointsIt);
    }

    while (m_renderedColors.size() < elements.size() * NumberTrianglesInTetrahedron * sofa::geometry::Triangle::NumberOfNodes)
    {
        static constexpr std::array colors = generateElementColors<NumberTrianglesInTetrahedron, sofa::geometry::Triangle::NumberOfNodes>({
            sofa::type::RGBAColor::blue(),
            sofa::type::RGBAColor::black(),
            sofa::type::RGBAColor::azure(),
            sofa::type::RGBAColor::cyan()});

        m_renderedColors.insert(m_renderedColors.end(), colors.begin(), colors.end());
    }

    drawTool->drawTriangles(m_renderedPoints, m_renderedColors);
}

void VisualMesh::drawHexahedra(helper::visual::DrawTool* drawTool)
{
    if (!l_topology)
        return;

    const auto& elements = l_topology->getHexahedra();
    const auto& facets = l_topology->getQuads();

    static constexpr std::size_t NumberQuadsInHexahedron = 6;
    m_renderedPoints.resize(elements.size() * NumberQuadsInHexahedron * sofa::geometry::Quad::NumberOfNodes);

    const auto elementSpace = d_elementSpace.getValue();
    const auto& positionAccessor = sofa::helper::getReadAccessor(d_position);

    auto pointsIt = m_renderedPoints.begin();

    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];
        const auto& facetsInElement = l_topology->getQuadsInHexahedron(i);
        assert(facetsInElement.size() == NumberQuadsInHexahedron);

        const sofa::type::Vec3 center = elementCenter(positionAccessor.ref(), element);

        setPoints(facetsInElement, facets, positionAccessor.ref(), center, elementSpace, pointsIt);
    }

    while (m_renderedColors.size() < elements.size() * NumberQuadsInHexahedron * sofa::geometry::Quad::NumberOfNodes)
    {
        static constexpr std::array colors = generateElementColors<NumberQuadsInHexahedron, sofa::geometry::Quad::NumberOfNodes>({
            sofa::type::RGBAColor(0.7f,0.7f,0.1f,1.f),
            sofa::type::RGBAColor(0.7f,0.0f,0.0f,1.f),
            sofa::type::RGBAColor(0.0f,0.7f,0.0f,1.f),
            sofa::type::RGBAColor(0.0f,0.0f,0.7f,1.f),
            sofa::type::RGBAColor(0.1f,0.7f,0.7f,1.f),
            sofa::type::RGBAColor(0.7f,0.1f,0.7f,1.f)});

        m_renderedColors.insert(m_renderedColors.end(), colors.begin(), colors.end());
    }

    drawTool->drawQuads(m_renderedPoints, m_renderedColors);
}

void VisualMesh::doDrawVisual(const core::visual::VisualParams* vparams)
{
    auto* drawTool = vparams->drawTool();

    vparams->drawTool()->disableLighting();

    const auto hasTetra = l_topology && !l_topology->getTetrahedra().empty();
    const auto hasHexa = l_topology && !l_topology->getHexahedra().empty();

    if (!hasTetra && !hasHexa)
    {
        drawTriangles(drawTool);
    }
    drawTetrahedra(drawTool);
    drawHexahedra(drawTool);
}

void VisualMesh::validateTopology()
{
    if (l_topology.empty())
    {
        msg_info() << "Link to Topology container should be set to ensure right behavior. First "
                      "Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (l_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << this->l_topology.getLinkedPath()
                    << ", nor in current context: " << this->getContext()->name
                    << ". Object must have a BaseMeshTopology. "
                    << "The list of available BaseMeshTopology components is: "
                    << sofa::core::ObjectFactory::getInstance()
                           ->listClassesDerivedFrom<sofa::core::topology::BaseMeshTopology>();
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

}  // namespace sofa::component::visual
