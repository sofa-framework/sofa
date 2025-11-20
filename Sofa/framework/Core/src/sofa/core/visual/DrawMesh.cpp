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
#include <sofa/core/visual/DrawMesh.h>

namespace sofa::core::visual
{

namespace
{

template<class ElementType>
sofa::type::Vec3 elementCenter(const type::vector<type::Vec3>& position, const ElementType& element)
{
    sofa::type::Vec3 center{};
    for (sofa::Size vId = 0; vId < element.size(); ++vId)
    {
        center += position[element[vId]];
    }
    center /= element.size();
    return center;
}

template<std::size_t NumberFacetsInElement, class FacetType>
void setPoints(
    const std::array<sofa::Index, NumberFacetsInElement> facetsInElement,
    const sofa::type::vector<FacetType>& facets,
    const type::vector<type::Vec3>& position,
    const type::Vec3& elementCenter,
    SReal elementSpace,
    sofa::type::vector< sofa::type::Vec3 >::iterator& pointsIt
    )
{
    for (const auto& facetId : facetsInElement)
    {
        const auto& facet = facets[facetId];
        for (const auto vId : facet)
        {
            *pointsIt++ = (position[vId] - elementCenter) * (1._sreal - elementSpace) + elementCenter;
        }
    }
}

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

void DrawMesh::drawTriangles(
    sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position,
    sofa::core::topology::BaseMeshTopology* topology)
{
    m_drawTriangleMesh.draw(drawTool, position, topology);
}

void DrawMesh::drawTetrahedra(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology)
{
    m_drawTetrahedronMesh.draw(drawTool, position, topology);
}

void DrawMesh::drawHexahedra(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology)
{
    m_drawHexahedronMesh.draw(drawTool, position, topology);
}

void DrawMesh::setElementSpace(SReal elementSpace)
{
    m_drawTriangleMesh.elementSpace =
    m_drawTetrahedronMesh.elementSpace =
    m_drawHexahedronMesh.elementSpace = elementSpace;
}

void DrawMesh::drawSurface(sofa::helper::visual::DrawTool* drawTool,
                           const type::vector<type::Vec3>& position,
                           sofa::core::topology::BaseMeshTopology* topology)
{
    drawTriangles(drawTool, position, topology);
}

void DrawMesh::drawVolume(sofa::helper::visual::DrawTool* drawTool,
                          const type::vector<type::Vec3>& position,
                          sofa::core::topology::BaseMeshTopology* topology)
{
    drawTetrahedra(drawTool, position, topology);
    drawHexahedra(drawTool, position, topology);
}

void DrawMesh::draw(sofa::helper::visual::DrawTool* drawTool,
                    const type::vector<type::Vec3>& position,
                    sofa::core::topology::BaseMeshTopology* topology)
{
    if (!topology)
    {
        return;
    }

    const auto hasTetra = topology && !topology->getTetrahedra().empty();
    const auto hasHexa = topology && !topology->getHexahedra().empty();

    if (!hasTetra && !hasHexa)
    {
        drawSurface(drawTool, position, topology);
    }
    drawVolume(drawTool, position, topology);
}

void BaseDrawMesh::draw(sofa::helper::visual::DrawTool* drawTool,
                        const type::vector<type::Vec3>& position,
                        sofa::core::topology::BaseMeshTopology* topology)
{
    if (!drawTool)
        return;
    if (!topology)
        return;

    const auto stateLifeCycle = drawTool->makeStateLifeCycle();
    drawTool->disableLighting();

    doDraw(drawTool, position, topology);
}

void DrawElementMesh<sofa::geometry::Triangle>::doDraw(
    sofa::helper::visual::DrawTool* drawTool,
    const type::vector<type::Vec3>& position,
    sofa::core::topology::BaseMeshTopology* topology)
{
    if (!topology)
        return;

    const auto& elements = topology->getTriangles();

    renderedPoints.resize(elements.size() * sofa::geometry::Triangle::NumberOfNodes);

    auto pointsIt = renderedPoints.begin();

    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];

        const sofa::type::Vec3 center = elementCenter(position, element);

        std::array facetsInElement{i};
        setPoints(facetsInElement, elements, position, center, elementSpace, pointsIt);
    }

    while (renderedColors.size() < elements.size() * sofa::geometry::Triangle::NumberOfNodes)
    {
        static constexpr std::array colors{
            type::makeHomogeneousArray<sofa::type::RGBAColor, sofa::geometry::Triangle::NumberOfNodes>( sofa::type::RGBAColor::green()),
            type::makeHomogeneousArray<sofa::type::RGBAColor, sofa::geometry::Triangle::NumberOfNodes>(sofa::type::RGBAColor::teal()),
            type::makeHomogeneousArray<sofa::type::RGBAColor, sofa::geometry::Triangle::NumberOfNodes>(sofa::type::RGBAColor::blue()),
        };

        const auto elementId = renderedColors.size() / sofa::geometry::Triangle::NumberOfNodes;
        const auto triangleColor = colors[elementId % colors.size()];

        renderedColors.insert(renderedColors.end(), triangleColor.begin(), triangleColor.end());
    }

    drawTool->drawTriangles(renderedPoints, renderedColors);
}

void DrawElementMesh<geometry::Tetrahedron>::doDraw(sofa::helper::visual::DrawTool* drawTool,
                                                  const type::vector<type::Vec3>& position,
                                                  sofa::core::topology::BaseMeshTopology* topology)
{
    if (!topology)
        return;

    const auto& elements = topology->getTetrahedra();
    const auto& facets = topology->getTriangles();

    static constexpr std::size_t NumberTrianglesInTetrahedron = 4;
    renderedPoints.resize(elements.size() * NumberTrianglesInTetrahedron * sofa::geometry::Triangle::NumberOfNodes);

    auto pointsIt = renderedPoints.begin();

    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];
        const auto& facetsInElement = topology->getTrianglesInTetrahedron(i);
        assert(facetsInElement.size() == NumberTrianglesInTetrahedron);

        const sofa::type::Vec3 center = elementCenter(position, element);

        setPoints(facetsInElement, facets, position, center, elementSpace, pointsIt);
    }

    while (renderedColors.size() < elements.size() * NumberTrianglesInTetrahedron * sofa::geometry::Triangle::NumberOfNodes)
    {
        static constexpr std::array colors = generateElementColors<NumberTrianglesInTetrahedron, sofa::geometry::Triangle::NumberOfNodes>({
            sofa::type::RGBAColor::blue(),
            sofa::type::RGBAColor::black(),
            sofa::type::RGBAColor::azure(),
            sofa::type::RGBAColor::cyan()});

        renderedColors.insert(renderedColors.end(), colors.begin(), colors.end());
    }

    drawTool->drawTriangles(renderedPoints, renderedColors);
}
void DrawElementMesh<geometry::Hexahedron>::doDraw(sofa::helper::visual::DrawTool* drawTool,
                                                 const type::vector<type::Vec3>& position,
                                                 sofa::core::topology::BaseMeshTopology* topology)
{
    if (!topology)
        return;

    const auto& elements = topology->getHexahedra();
    const auto& facets = topology->getQuads();

    static constexpr std::size_t NumberQuadsInHexahedron = 6;
    renderedPoints.resize(elements.size() * NumberQuadsInHexahedron * sofa::geometry::Quad::NumberOfNodes);

    auto pointsIt = renderedPoints.begin();

    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];
        const auto& facetsInElement = topology->getQuadsInHexahedron(i);
        assert(facetsInElement.size() == NumberQuadsInHexahedron);

        const sofa::type::Vec3 center = elementCenter(position, element);

        setPoints(facetsInElement, facets, position, center, elementSpace, pointsIt);
    }

    while (renderedColors.size() < elements.size() * NumberQuadsInHexahedron * sofa::geometry::Quad::NumberOfNodes)
    {
        static constexpr std::array colors = generateElementColors<NumberQuadsInHexahedron, sofa::geometry::Quad::NumberOfNodes>({
            sofa::type::RGBAColor(0.7f,0.7f,0.1f,1.f),
            sofa::type::RGBAColor(0.7f,0.0f,0.0f,1.f),
            sofa::type::RGBAColor(0.0f,0.7f,0.0f,1.f),
            sofa::type::RGBAColor(0.0f,0.0f,0.7f,1.f),
            sofa::type::RGBAColor(0.1f,0.7f,0.7f,1.f),
            sofa::type::RGBAColor(0.7f,0.1f,0.7f,1.f)});

        renderedColors.insert(renderedColors.end(), colors.begin(), colors.end());
    }

    drawTool->drawQuads(renderedPoints, renderedColors);
}

}  // namespace sofa::core::visual
