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

}  // namespace



void DrawMesh::drawTriangles(
    sofa::helper::visual::DrawTool* drawTool,
    const type::vector<type::Vec3>& position,
    sofa::core::topology::BaseMeshTopology* topology)
{
    m_drawTriangleMesh.draw(drawTool, position, topology);
}

void DrawMesh::drawTetrahedra(
    sofa::helper::visual::DrawTool* drawTool,
    const type::vector<type::Vec3>& position,
    sofa::core::topology::BaseMeshTopology* topology)
{
    m_drawTetrahedronMesh.draw(drawTool, position, topology);
}

void DrawMesh::drawHexahedra(
    sofa::helper::visual::DrawTool* drawTool,
    const type::vector<type::Vec3>& position,
    sofa::core::topology::BaseMeshTopology* topology)
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

void DrawElementMesh<geometry::Triangle>::doDraw(
    sofa::helper::visual::DrawTool* drawTool,
   const type::vector<type::Vec3>& position,
   sofa::core::topology::BaseMeshTopology* topology,
   const ColorContainer& colors)
{
    if (!topology)
        return;

    const auto& elements = topology->getTriangles();

    const auto size = (elements.size() / NumberColors) * sofa::geometry::Triangle::NumberOfNodes;
    for ( auto& p : renderedPoints)
    {
        p.resize(size);
    }

    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];

        const sofa::type::Vec3 center = elementCenter(position, element);

        for (std::size_t j = 0; j < sofa::geometry::Triangle::NumberOfNodes; ++j)
        {
            renderedPoints[i % NumberColors][i / NumberColors + j] = applyElementSpace(position[element[j]], center);
        }
    }

    for (std::size_t j = 0; j < NumberColors; ++j)
    {
        drawTool->drawTriangles(renderedPoints[j], colors[j]);
    }
}

void DrawElementMesh<geometry::Tetrahedron>::doDraw(
    sofa::helper::visual::DrawTool* drawTool,
    const type::vector<type::Vec3>& position,
    sofa::core::topology::BaseMeshTopology* topology,
    const ColorContainer& colors)
{
    if (!topology)
        return;

    const auto& elements = topology->getTetrahedra();
    const auto& facets = topology->getTriangles();

    for ( auto& p : renderedPoints)
    {
        p.resize(elements.size() * sofa::geometry::Triangle::NumberOfNodes);
    }

    std::size_t renderedPointId {};
    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];
        const auto& facetsInElement = topology->getTrianglesInTetrahedron(i);
        assert(facetsInElement.size() == NumberTrianglesInTetrahedron);

        const sofa::type::Vec3 center = elementCenter(position, element);

        for (std::size_t j = 0; j < NumberTrianglesInTetrahedron; ++j)
        {
            const auto faceId = facetsInElement[j];
            for (const auto vertexId : facets[faceId])
            {
                renderedPoints[j][renderedPointId++] = applyElementSpace(position[vertexId], center);
            }
        }
    }

    for (std::size_t j = 0; j < NumberTrianglesInTetrahedron; ++j)
    {
        drawTool->drawTriangles(renderedPoints[j], colors[j]);
    }
}

void DrawElementMesh<geometry::Hexahedron>::doDraw(sofa::helper::visual::DrawTool* drawTool,
                                                   const type::vector<type::Vec3>& position,
                                                   sofa::core::topology::BaseMeshTopology* topology,
                                                   const ColorContainer& colors)
{
    if (!topology)
        return;

    const auto& elements = topology->getHexahedra();
    const auto& facets = topology->getQuads();

    for ( auto& p : renderedPoints)
    {
        p.resize(elements.size() * sofa::geometry::Quad::NumberOfNodes);
    }

    std::size_t renderedPointId {};
    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];
        const auto& facetsInElement = topology->getQuadsInHexahedron(i);
        assert(facetsInElement.size() == NumberTrianglesInTetrahedron);

        const sofa::type::Vec3 center = elementCenter(position, element);

        for (std::size_t j = 0; j < NumberQuadsInHexahedron; ++j)
        {
            const auto faceId = facetsInElement[j];
            for (const auto vertexId : facets[faceId])
            {
                renderedPoints[j][renderedPointId++] = applyElementSpace(position[vertexId], center);
            }
        }
    }

    for (std::size_t j = 0; j < NumberQuadsInHexahedron; ++j)
    {
        drawTool->drawTriangles(renderedPoints[j], colors[j]);
    }
}

}  // namespace sofa::core::visual
