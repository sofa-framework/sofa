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

#include <sofa/core/config.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/helper/IotaView.h>
#include <sofa/helper/visual/DrawTool.h>
#include <sofa/topology/QuadraticElements.h>

#include <ranges>

namespace sofa::core::visual
{

/**
 * @class BaseDrawMesh
 * @brief Base class for mesh drawing using Curiously Recurring Template Pattern (CRTP)
 */
template<class Derived, std::size_t NumberColors_>
struct BaseDrawMesh
{
    static constexpr std::size_t NumberColors { NumberColors_ };
    using ColorContainer = std::array<sofa::type::RGBAColor, NumberColors>;

    SReal elementSpace { 0.125_sreal };

    /**
     * @brief Draws all elements of the mesh using provided colors.
     *
     * This function draws all elements of the mesh (from index 0 to the total number of elements)
     * using the specified colors. It serves as a convenience wrapper that calls \ref
     * drawSomeElements with all element indices.
     *
     * @tparam PositionContainer The type of the position container (e.g.,
     * sofa::type::vector<sofa::type::Vec3>)
     * @param drawTool The drawing tool to use for rendering
     * @param position The container of vertex positions
     * @param topology The mesh topology
     * @param colors The colors to use for each element. If not specified, the default colors of the
     * derived class are used.
     */
    template<class PositionContainer>
    void drawAllElements(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const ColorContainer& colors = Derived::defaultColors)
    {
        const auto totalNbElements = topology->getElements<typename Derived::ElementType>().size();
        const auto elementsToDraw = sofa::helper::IotaView(static_cast<decltype(totalNbElements)>(0), totalNbElements);
        drawSomeElements(drawTool, position, topology, elementsToDraw, colors);
    }

    /**
     * @brief Draws a subset of elements of the mesh using provided colors.
     *
     * This function draws a specific subset of elements (specified by element indices) using the
     * provided colors.
     *
     * @tparam PositionContainer The type of the position container (e.g.,
     * sofa::type::vector<sofa::type::Vec3>)
     * @tparam IndicesContainer The type of container holding element indices (e.g.,
     * sofa::type::vector<sofa::Index>)
     * @param drawTool The drawing tool to use for rendering
     * @param position The container of vertex positions
     * @param topology The mesh topology
     * @param elementIndices The indices of elements to draw
     * @param colors The colors to use for each element. If not specified, the default colors of the
     * derived class are used.
     */
    template<class PositionContainer, class IndicesContainer>
    void drawSomeElements(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors = Derived::defaultColors)
    {
        if (!drawTool)
            return;
        if (!topology)
            return;

        static_cast<Derived&>(*this).doDraw(drawTool, position, topology, elementIndices, colors);
    }

protected:
    template<class PositionType>
    PositionType applyElementSpace(const PositionType& position, const PositionType& elementCenter) const
    {
        return (position - elementCenter) * (1._sreal - elementSpace) + elementCenter;
    }

    template<class PositionContainer, class ElementType>
    static typename PositionContainer::value_type elementCenter(const PositionContainer& position, const ElementType& element)
    {
        typename PositionContainer::value_type center{};
        for (sofa::Size vId = 0; vId < element.size(); ++vId)
        {
            center += position[element[vId]];
        }
        center /= element.size();
        return center;
    }

    /**
     * @brief Pre-allocated point buffers for rendering different color channels
     *
     * This static array holds pre-allocated vertex position buffers for each color channel.
     *
     * Key characteristics:
     * - Each buffer corresponds to one of the `NumberColors` color channels (e.g., 3 for triangles)
     * - Designed for efficient reuse across multiple drawing calls without reallocation
     */
    std::array<sofa::type::vector< sofa::type::Vec3 >, NumberColors> renderedPoints;
};

template<class ElementType>
struct DrawElementMesh{};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Edge>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::Edge>, 2>
{
    using ElementType = sofa::geometry::Edge;
    friend BaseDrawMesh;
    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::lime(),
        sofa::type::RGBAColor::silver()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getEdges();

        const auto size = elementIndices.size() * sofa::geometry::Edge::NumberOfNodes;
        for ( auto& p : renderedPoints)
        {
            p.resize(size);
        }

        std::array<std::size_t, NumberColors> renderedPointId {};
        for (auto i : elementIndices)
        {
            const auto& element = elements[i];

            const auto center = this->elementCenter(position, element);

            for (std::size_t j = 0; j < sofa::geometry::Edge::NumberOfNodes; ++j)
            {
                const auto p = this->applyElementSpace(position[element[j]], center);
                renderedPoints[i % NumberColors][renderedPointId[i%NumberColors]++] = sofa::type::toVec3(p);
            }
        }

        for (std::size_t j = 0; j < NumberColors; ++j)
        {
            drawTool->drawLines(renderedPoints[j], 2.f, colors[j]);
        }
    }
};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Triangle>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::Triangle>, 3>
{
    using ElementType = sofa::geometry::Triangle;
    friend BaseDrawMesh;
    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::green(),
        sofa::type::RGBAColor::teal(),
        sofa::type::RGBAColor::blue()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getTriangles();

        const auto size = (elementIndices.size() / NumberColors + 1) * sofa::geometry::Triangle::NumberOfNodes;
        for ( auto& p : renderedPoints)
        {
            p.resize(size);
        }

        std::array<std::size_t, NumberColors> renderedPointId {};
        for (auto i : elementIndices)
        {
            const auto& element = elements[i];

            const auto center = this->elementCenter(position, element);

            for (std::size_t j = 0; j < sofa::geometry::Triangle::NumberOfNodes; ++j)
            {
                const auto p = this->applyElementSpace(position[element[j]], center);
                renderedPoints[i % NumberColors][renderedPointId[i%NumberColors]++] = sofa::type::toVec3(p);
            }
        }

        for (std::size_t j = 0; j < NumberColors; ++j)
        {
            drawTool->drawTriangles(renderedPoints[j], colors[j]);
        }
    }
};


template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Quad>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::Quad>, 2>
{
    using ElementType = sofa::geometry::Quad;
    friend BaseDrawMesh;
    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::green(),
        sofa::type::RGBAColor::orange()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getQuads();

        const auto size = elementIndices.size() * sofa::geometry::Quad::NumberOfNodes;
        for ( auto& p : renderedPoints)
        {
            p.resize(size);
        }

        std::array<std::size_t, NumberColors> renderedPointId {};
        for (auto i : elementIndices)
        {
            const auto& element = elements[i];

            const auto center = this->elementCenter(position, element);

            for (std::size_t j = 0; j < sofa::geometry::Quad::NumberOfNodes; ++j)
            {
                const auto p = this->applyElementSpace(position[element[j]], center);
                renderedPoints[i % NumberColors][renderedPointId[i%NumberColors]++] = sofa::type::toVec3(p);
            }
        }

        for (std::size_t j = 0; j < NumberColors; ++j)
        {
            drawTool->drawQuads(renderedPoints[j], colors[j]);
        }
    }
};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Tetrahedron>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::Tetrahedron>, 4>
{
    using ElementType = sofa::geometry::Tetrahedron;
    friend BaseDrawMesh;
    static constexpr std::size_t NumberTrianglesInTetrahedron = 4;

    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::blue(),
        sofa::type::RGBAColor::black(),
        sofa::type::RGBAColor::azure(),
        sofa::type::RGBAColor::cyan()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getTetrahedra();
        const auto& facets = topology->getTriangles();

        if(facets.empty())
        {
            msg_error_once("DrawElementMesh<Tetrahedron>") << "Drawing tetrahedra needs the associated triangles in the topology.";
            return;
        }

        for ( auto& p : renderedPoints)
        {
            p.resize(elementIndices.size() * sofa::geometry::Triangle::NumberOfNodes);
        }

        for (auto i : elementIndices)
        {
            const auto& element = elements[i];
            const auto& facetsInElement = topology->getTrianglesInTetrahedron(i);
            assert(facetsInElement.size() == NumberTrianglesInTetrahedron);

            const auto center = this->elementCenter(position, element);

            for (std::size_t j = 0; j < NumberTrianglesInTetrahedron; ++j)
            {
                const auto faceId = facetsInElement[j];
                std::size_t k {};
                for (const auto vertexId : facets[faceId])
                {
                    const auto p = this->applyElementSpace(position[vertexId], center);
                    renderedPoints[j][i * sofa::geometry::Triangle::NumberOfNodes + k] = sofa::type::toVec3(p);
                    ++k;
                }
            }
        }

        for (std::size_t j = 0; j < NumberTrianglesInTetrahedron; ++j)
        {
            drawTool->drawTriangles(renderedPoints[j], colors[j]);
        }
    }
};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Prism>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::Prism>, 5>
{
    using ElementType = sofa::geometry::Prism;
    friend BaseDrawMesh;
    static constexpr std::size_t NumberTrianglesInPrism = 2;
    static constexpr std::size_t NumberQuadsInPrism = 3;

    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::green(),
        sofa::type::RGBAColor::teal(),
        sofa::type::RGBAColor::navy(),
        sofa::type::RGBAColor::gold(),
        sofa::type::RGBAColor::purple()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getPrisms();

        // Allocate space for rendering points
        renderedPoints[0].resize(elementIndices.size() * sofa::geometry::Triangle::NumberOfNodes);
        renderedPoints[1].resize(elementIndices.size() * sofa::geometry::Triangle::NumberOfNodes);
        renderedPoints[2].resize(elementIndices.size() * sofa::geometry::Quad::NumberOfNodes);
        renderedPoints[3].resize(elementIndices.size() * sofa::geometry::Quad::NumberOfNodes);
        renderedPoints[4].resize(elementIndices.size() * sofa::geometry::Quad::NumberOfNodes);

        std::array<std::size_t, NumberColors> renderedPointId {};
        
        for (auto i : elementIndices)
        {
            const auto& prism = elements[i];
            const auto center = this->elementCenter(position, prism);

            const auto drawTriangle = [&](sofa::Index bufferId, sofa::Index v0, sofa::Index v1, sofa::Index v2)
            {
                const std::array vertexIndices { prism[v0], prism[v1], prism[v2] };
                for (std::size_t k = 0; k < sofa::geometry::Triangle::NumberOfNodes; ++k)
                {
                    const auto p = this->applyElementSpace(position[vertexIndices[k]], center);
                    renderedPoints[bufferId][renderedPointId[bufferId]++] = sofa::type::toVec3(p);
                }
            };

            drawTriangle(0, 0, 1, 2);
            drawTriangle(1, 5, 4, 3);

            const auto drawQuad = [&](sofa::Index bufferId, sofa::Index v0, sofa::Index v1, sofa::Index v2, sofa::Index v3)
            {
                const std::array vertexIndices { prism[v0], prism[v1], prism[v2], prism[v3] };
                for (std::size_t k = 0; k < sofa::geometry::Quad::NumberOfNodes; ++k)
                {
                    const auto p = this->applyElementSpace(position[vertexIndices[k]], center);
                    renderedPoints[bufferId][renderedPointId[bufferId]++] = sofa::type::toVec3(p);
                }
            };

            drawQuad(2, 0, 2, 5, 3);
            drawQuad(3, 0, 3, 4, 1);
            drawQuad(4, 1, 4, 5, 2);
        }

        drawTool->drawTriangles(renderedPoints[0], colors[0]);
        drawTool->drawTriangles(renderedPoints[1], colors[1]);
        drawTool->drawQuads(renderedPoints[2], colors[2]);
        drawTool->drawQuads(renderedPoints[3], colors[3]);
        drawTool->drawQuads(renderedPoints[4], colors[4]);
    }
};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Pyramid>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::Pyramid>, 5>
{
    using ElementType = sofa::geometry::Pyramid;
    friend BaseDrawMesh;

    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::green(),
        sofa::type::RGBAColor::teal(),
        sofa::type::RGBAColor::navy(),
        sofa::type::RGBAColor::gold(),
        sofa::type::RGBAColor::purple()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getPyramids();

        // Allocate space for rendering points
        // 1 Quad + 4 Triangles
        renderedPoints[0].resize(elementIndices.size() * sofa::geometry::Quad::NumberOfNodes);
        renderedPoints[1].resize(elementIndices.size() * sofa::geometry::Triangle::NumberOfNodes);
        renderedPoints[2].resize(elementIndices.size() * sofa::geometry::Triangle::NumberOfNodes);
        renderedPoints[3].resize(elementIndices.size() * sofa::geometry::Triangle::NumberOfNodes);
        renderedPoints[4].resize(elementIndices.size() * sofa::geometry::Triangle::NumberOfNodes);

        std::array<std::size_t, NumberColors> renderedPointId {};

        for (auto i : elementIndices)
        {
            const auto& pyramid = elements[i];
            const auto center = this->elementCenter(position, pyramid);

            const auto drawQuad = [&](sofa::Index bufferId, sofa::Index v0, sofa::Index v1, sofa::Index v2, sofa::Index v3)
            {
                const std::array vertexIndices { pyramid[v0], pyramid[v1], pyramid[v2], pyramid[v3] };
                for (std::size_t k = 0; k < sofa::geometry::Quad::NumberOfNodes; ++k)
                {
                    const auto p = this->applyElementSpace(position[vertexIndices[k]], center);
                    renderedPoints[bufferId][renderedPointId[bufferId]++] = sofa::type::toVec3(p);
                }
            };

            const auto drawTriangle = [&](sofa::Index bufferId, sofa::Index v0, sofa::Index v1, sofa::Index v2)
            {
                const std::array vertexIndices { pyramid[v0], pyramid[v1], pyramid[v2] };
                for (std::size_t k = 0; k < sofa::geometry::Triangle::NumberOfNodes; ++k)
                {
                    const auto p = this->applyElementSpace(position[vertexIndices[k]], center);
                    renderedPoints[bufferId][renderedPointId[bufferId]++] = sofa::type::toVec3(p);
                }
            };

            drawQuad(0, 0, 3, 2, 1);
            drawTriangle(1, 0, 1, 4);
            drawTriangle(2, 1, 2, 4);
            drawTriangle(3, 3, 4, 2);
            drawTriangle(4, 0, 4, 3);
        }

        drawTool->drawQuads(renderedPoints[0], colors[0]);
        drawTool->drawTriangles(renderedPoints[1], colors[1]);
        drawTool->drawTriangles(renderedPoints[2], colors[2]);
        drawTool->drawTriangles(renderedPoints[3], colors[3]);
        drawTool->drawTriangles(renderedPoints[4], colors[4]);
    }
};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Hexahedron>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::Hexahedron>, 6>
{
    using ElementType = sofa::geometry::Hexahedron;
    friend BaseDrawMesh;
    static constexpr std::size_t NumberQuadsInHexahedron = 6;

    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor(0.7f,0.7f,0.1f,1.f),
        sofa::type::RGBAColor(0.7f,0.0f,0.0f,1.f),
        sofa::type::RGBAColor(0.0f,0.7f,0.0f,1.f),
        sofa::type::RGBAColor(0.0f,0.0f,0.7f,1.f),
        sofa::type::RGBAColor(0.1f,0.7f,0.7f,1.f),
        sofa::type::RGBAColor(0.7f,0.1f,0.7f,1.f)
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getHexahedra();
        const auto& facets = topology->getQuads();

        if(facets.empty())
        {
            msg_error_once("DrawElementMesh<Hexahedron>") << "Drawing hexahedra needs the associated quads in the topology.";
            return;
        }

        for ( auto& p : renderedPoints)
        {
            p.resize(elementIndices.size() * sofa::geometry::Quad::NumberOfNodes);
        }

        for (auto i : elementIndices)
        {
            const auto& element = elements[i];
            const auto& facetsInElement = topology->getQuadsInHexahedron(i);
            assert(facetsInElement.size() == NumberQuadsInHexahedron);

            const auto center = this->elementCenter(position, element);

            for (std::size_t j = 0; j < NumberQuadsInHexahedron; ++j)
            {
                const auto faceId = facetsInElement[j];
                std::size_t k {};
                for (const auto vertexId : facets[faceId])
                {
                    const auto p = this->applyElementSpace(position[vertexId], center);
                    renderedPoints[j][i * sofa::geometry::Quad::NumberOfNodes + k] = sofa::type::toVec3(p);
                    ++k;
                }
            }
        }

        for (std::size_t j = 0; j < NumberQuadsInHexahedron; ++j)
        {
            drawTool->drawQuads(renderedPoints[j], colors[j]);
        }
    }
};


template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::QuadraticEdge>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::QuadraticEdge>, 2>
{
    using ElementType = sofa::geometry::QuadraticEdge;
    friend BaseDrawMesh;
    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::lime(),
        sofa::type::RGBAColor::silver()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getElements<sofa::geometry::QuadraticEdge>();

        // Draw as 2 line segments per quadratic edge
        constexpr std::size_t nbSegments = 2;
        const auto size = elementIndices.size() * nbSegments * 2;
        for (auto& p : renderedPoints)
        {
            p.resize(size);
        }

        std::array<std::size_t, NumberColors> renderedPointId {};
        for (auto i : elementIndices)
        {
            const auto& element = elements[i];
            const auto center = this->elementCenter(position, sofa::topology::Edge{element[0], element[1]});

            // Segment 0-4 (vertex 0 to mid-edge)
            renderedPoints[i % NumberColors][renderedPointId[i % NumberColors]++] =
                sofa::type::toVec3(this->applyElementSpace(position[element[0]], center));
            renderedPoints[i % NumberColors][renderedPointId[i % NumberColors]++] =
                sofa::type::toVec3(this->applyElementSpace(position[element[2]], center));

            // Segment 4-1 (mid-edge to vertex 1)
            renderedPoints[i % NumberColors][renderedPointId[i % NumberColors]++] =
                sofa::type::toVec3(this->applyElementSpace(position[element[2]], center));
            renderedPoints[i % NumberColors][renderedPointId[i % NumberColors]++] =
                sofa::type::toVec3(this->applyElementSpace(position[element[1]], center));
        }

        for (std::size_t j = 0; j < NumberColors; ++j)
        {
            drawTool->drawLines(renderedPoints[j], 2.f, colors[j]);
        }
    }
};


template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::QuadraticTriangle>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::QuadraticTriangle>, 3>
{
    using ElementType = sofa::geometry::QuadraticTriangle;
    friend BaseDrawMesh;
    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::green(),
        sofa::type::RGBAColor::teal(),
        sofa::type::RGBAColor::blue()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getElements<sofa::geometry::QuadraticTriangle>();

        // Each quadratic triangle is subdivided into 4 triangles
        constexpr std::size_t nbSubTriangles = 4;
        for (auto& p : renderedPoints)
        {
            p.resize(elementIndices.size() * nbSubTriangles * sofa::geometry::Triangle::NumberOfNodes);
        }

        // Sub-triangles: corner triangles + center triangle
        static constexpr std::array<std::array<std::size_t, 3>, 4> subTriangles {{
            {{0, 3, 5}},  // corner at vertex 0
            {{3, 1, 4}},  // corner at vertex 1
            {{5, 4, 2}},  // corner at vertex 2
            {{3, 4, 5}}   // center triangle
        }};

        for (std::size_t i = 0; i < elementIndices.size(); ++i)
        {
            const auto& element = elements[elementIndices[i]];
            const auto center = this->elementCenter(position, sofa::topology::Triangle{element[0], element[1], element[2]});

            for (std::size_t t = 0; t < nbSubTriangles; ++t)
            {
                for (std::size_t v = 0; v < 3; ++v)
                {
                    const auto vertexId = element[subTriangles[t][v]];
                    const auto p = this->applyElementSpace(position[vertexId], center);
                    renderedPoints[(i + t) % NumberColors][(i * nbSubTriangles + t) * 3 + v] = sofa::type::toVec3(p);
                }
            }
        }

        for (std::size_t j = 0; j < NumberColors; ++j)
        {
            drawTool->drawTriangles(renderedPoints[j], colors[j]);
        }
    }
};


template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::QuadraticQuad>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::QuadraticQuad>, 2>
{
    using ElementType = sofa::geometry::QuadraticQuad;
    friend BaseDrawMesh;
    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::green(),
        sofa::type::RGBAColor::orange()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getElements<sofa::geometry::QuadraticQuad>();

        // Each quadratic quad is subdivided into 4 quads
        constexpr std::size_t nbSubQuads = 4;
        const auto size = elementIndices.size() * nbSubQuads * sofa::geometry::Quad::NumberOfNodes;
        for (auto& p : renderedPoints)
        {
            p.resize(size);
        }

        // Sub-quads using corner vertices, mid-edges, and center
        static constexpr std::array<std::array<std::size_t, 4>, 4> subQuads {{
            {{0, 4, 8, 7}},  // bottom-left
            {{4, 1, 5, 8}},  // bottom-right
            {{8, 5, 2, 6}},  // top-right
            {{7, 8, 6, 3}}   // top-left
        }};

        std::array<std::size_t, NumberColors> renderedPointId {};
        for (auto i : elementIndices)
        {
            const auto& element = elements[i];
            const auto center = this->elementCenter(position, sofa::topology::Quad{element[0], element[1], element[2], element[3]});

            for (std::size_t q = 0; q < nbSubQuads; ++q)
            {
                for (std::size_t v = 0; v < 4; ++v)
                {
                    const auto vertexId = element[subQuads[q][v]];
                    const auto p = this->applyElementSpace(position[vertexId], center);
                    renderedPoints[i % NumberColors][renderedPointId[i % NumberColors]++] = sofa::type::toVec3(p);
                }
            }
        }

        for (std::size_t j = 0; j < NumberColors; ++j)
        {
            drawTool->drawQuads(renderedPoints[j], colors[j]);
        }
    }
};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::QuadraticTetrahedron>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::QuadraticTetrahedron>, 4>
{
    using ElementType = sofa::geometry::QuadraticTetrahedron;
    friend BaseDrawMesh;

    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::blue(),
        sofa::type::RGBAColor::black(),
        sofa::type::RGBAColor::azure(),
        sofa::type::RGBAColor::cyan()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getElements<sofa::geometry::QuadraticTetrahedron>();

        // Each QuadraticTetrahedron has 4 quadratic faces.
        // Each quadratic face is drawn as 4 triangles.
        constexpr std::size_t nbTrianglesPerFace = 4;
        constexpr std::size_t nbFaces = 4;

        for (auto& p : renderedPoints)
        {
            p.resize(elementIndices.size() * nbTrianglesPerFace * sofa::geometry::Triangle::NumberOfNodes);
        }

        for (std::size_t i = 0; i < elementIndices.size(); ++i)
        {
            const auto& element = elements[elementIndices[i]];
            const auto center = this->elementCenter(position, sofa::topology::Tetrahedron{element[0], element[1], element[2], element[3]});

            for (std::size_t f = 0; f < nbFaces; ++f)
            {
                const auto& face = topology::quadraticTrianglesInQuadraticTetrahedronArray[f];
                const sofa::topology::QuadraticTriangle FACE(
                    element[face[0]], element[face[1]], element[face[2]],
                    element[face[3]], element[face[4]], element[face[5]]);

                for (std::size_t t = 0; t < nbTrianglesPerFace; ++t)
                {
                    const auto& triangle = topology::trianglesInQuadraticTriangles[t];
                    const sofa::topology::Triangle TRIANGLE(FACE[triangle[0]], FACE[triangle[1]], FACE[triangle[2]]);

                    for (std::size_t v = 0; v < 3; ++v)
                    {
                        const auto vertexId = TRIANGLE[v];
                        const auto p = this->applyElementSpace(position[vertexId], center);
                        renderedPoints[f][(i * nbTrianglesPerFace + t) * 3 + v] = sofa::type::toVec3(p);
                    }
                }
            }
        }

        for (std::size_t f = 0; f < nbFaces; ++f)
        {
            // Generate slightly different colors for the 4 sub-triangles of a face
            std::array<sofa::type::RGBAColor, 4> subColors;
            for (int t = 0; t < 4; ++t)
            {
                subColors[t] = type::RGBAColor::lighten(colors[f], t * 0.15_sreal);
            }

            // Draw each sub-triangle set with its specific color
            for (std::size_t t = 0; t < nbTrianglesPerFace; ++t)
            {
                // We need a temporary view of the buffer for this specific sub-triangle across all tetrahedra
                sofa::type::vector<sofa::type::Vec3> subBuffer;
                subBuffer.resize(elementIndices.size() * 3);
                for(std::size_t i = 0; i < elementIndices.size(); ++i)
                {
                    subBuffer[i*3 + 0] = renderedPoints[f][(i * nbTrianglesPerFace + t) * 3 + 0];
                    subBuffer[i*3 + 1] = renderedPoints[f][(i * nbTrianglesPerFace + t) * 3 + 1];
                    subBuffer[i*3 + 2] = renderedPoints[f][(i * nbTrianglesPerFace + t) * 3 + 2];
                }
                drawTool->drawTriangles(subBuffer, subColors[t]);
            }
        }
    }
};


template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::QuadraticHexahedron>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::QuadraticHexahedron>, 6>
{
    using ElementType = sofa::geometry::QuadraticHexahedron;
    friend BaseDrawMesh;
    static constexpr std::size_t NumberQuadsInHexahedron = 6;

    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor(0.7f,0.7f,0.1f,1.f),
        sofa::type::RGBAColor(0.7f,0.0f,0.0f,1.f),
        sofa::type::RGBAColor(0.0f,0.7f,0.0f,1.f),
        sofa::type::RGBAColor(0.0f,0.0f,0.7f,1.f),
        sofa::type::RGBAColor(0.1f,0.7f,0.7f,1.f),
        sofa::type::RGBAColor(0.7f,0.1f,0.7f,1.f)
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getElements<sofa::geometry::QuadraticHexahedron>();

        // Each face (6 total) is subdivided into 4 quads
        constexpr std::size_t nbQuadsPerFace = 4;
        for (auto& p : renderedPoints)
        {
            p.resize(elementIndices.size() * nbQuadsPerFace * sofa::geometry::Quad::NumberOfNodes);
        }

        for (std::size_t i = 0; i < elementIndices.size(); ++i)
        {
            const auto& element = elements[elementIndices[i]];
            const auto center = this->elementCenter(position,
                sofa::topology::Hexahedron{element[0], element[1], element[2], element[3], element[4], element[5], element[6], element[7]});

            for (std::size_t f = 0; f < NumberQuadsInHexahedron; ++f)
            {
                const auto& faceIndices = topology::quadraticQuadsInQuadraticHexahedronArray[f];

                for (std::size_t q = 0; q < nbQuadsPerFace; ++q)
                {
                    const auto& subQuad = topology::quadsInQuadraticQuads[q];
                    for (std::size_t v = 0; v < 4; ++v)
                    {
                        const auto vertexId = element[faceIndices[subQuad[v]]];
                        const auto p = this->applyElementSpace(position[vertexId], center);
                        renderedPoints[f][(i * nbQuadsPerFace + q) * 4 + v] = sofa::type::toVec3(p);
                    }
                }
            }
        }

        for (std::size_t f = 0; f < NumberQuadsInHexahedron; ++f)
        {
            drawTool->drawQuads(renderedPoints[f], colors[f]);
        }
    }
};


template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::QuadraticPrism>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::QuadraticPrism>, 5>
{
    using ElementType = sofa::geometry::QuadraticPrism;
    friend BaseDrawMesh;

    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::green(),
        sofa::type::RGBAColor::teal(),
        sofa::type::RGBAColor::navy(),
        sofa::type::RGBAColor::gold(),
        sofa::type::RGBAColor::purple()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getElements<sofa::geometry::QuadraticPrism>();

        // 2 triangular faces (each subdivided into 4 triangles) + 3 quad faces (each subdivided into 4 quads)
        constexpr std::size_t nbSubTriangles = 4;
        constexpr std::size_t nbSubQuads = 4;

        renderedPoints[0].resize(elementIndices.size() * nbSubTriangles * 3);
        renderedPoints[1].resize(elementIndices.size() * nbSubTriangles * 3);
        renderedPoints[2].resize(elementIndices.size() * nbSubQuads * 4);
        renderedPoints[3].resize(elementIndices.size() * nbSubQuads * 4);
        renderedPoints[4].resize(elementIndices.size() * nbSubQuads * 4);

        // Bottom triangle (0,1,2) subdivided
        static constexpr std::array<std::array<std::size_t, 3>, 4> bottomTriangles {{
            {{0, 6, 8}}, {{6, 1, 7}}, {{8, 7, 2}}, {{6, 7, 8}}
        }};
        // Top triangle (3,4,5) subdivided
        static constexpr std::array<std::array<std::size_t, 3>, 4> topTriangles {{
            {{3, 12, 14}}, {{12, 4, 13}}, {{14, 13, 5}}, {{12, 13, 14}}
        }};
        // Quad faces subdivided (using corner vertices, mid-edges, and face centers)
        static constexpr std::array<std::array<std::size_t, 4>, 4> quadSubdivision {{
            {{0, 6, 15, 9}}, {{6, 1, 10, 15}}, {{15, 10, 4, 12}}, {{9, 15, 12, 3}}
        }};

        std::array<std::size_t, 5> pointId{};

        for (auto idx : elementIndices)
        {
            const auto& element = elements[idx];
            const auto center = this->elementCenter(position,
                sofa::topology::Prism{element[0], element[1], element[2], element[3], element[4], element[5]});

            // Bottom triangle
            for (const auto& tri : bottomTriangles)
            {
                for (std::size_t v = 0; v < 3; ++v)
                {
                    const auto p = this->applyElementSpace(position[element[tri[v]]], center);
                    renderedPoints[0][pointId[0]++] = sofa::type::toVec3(p);
                }
            }

            // Top triangle
            for (const auto& tri : topTriangles)
            {
                for (std::size_t v = 0; v < 3; ++v)
                {
                    const auto p = this->applyElementSpace(position[element[tri[v]]], center);
                    renderedPoints[1][pointId[1]++] = sofa::type::toVec3(p);
                }
            }

            // Three quad faces (0-1-4-3), (1-2-5-4), (2-0-3-5)
            std::array<std::array<std::size_t, 9>, 3> quadFaceNodes {{
                {{0, 1, 4, 3, 6, 10, 12, 9, 15}},    // face 0-1-4-3
                {{1, 2, 5, 4, 7, 11, 13, 10, 16}},   // face 1-2-5-4
                {{2, 0, 3, 5, 8, 9, 14, 11, 17}}     // face 2-0-3-5
            }};

            for (std::size_t faceIdx = 0; faceIdx < 3; ++faceIdx)
            {
                const auto& faceNodes = quadFaceNodes[faceIdx];
                for (const auto& quad : quadSubdivision)
                {
                    for (std::size_t v = 0; v < 4; ++v)
                    {
                        const auto localIdx = quad[v];
                        const auto vertexId = element[faceNodes[localIdx]];
                        const auto p = this->applyElementSpace(position[vertexId], center);
                        renderedPoints[2 + faceIdx][pointId[2 + faceIdx]++] = sofa::type::toVec3(p);
                    }
                }
            }
        }

        drawTool->drawTriangles(renderedPoints[0], colors[0]);
        drawTool->drawTriangles(renderedPoints[1], colors[1]);
        drawTool->drawQuads(renderedPoints[2], colors[2]);
        drawTool->drawQuads(renderedPoints[3], colors[3]);
        drawTool->drawQuads(renderedPoints[4], colors[4]);
    }
};


template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::QuadraticPyramid>
    : public BaseDrawMesh<DrawElementMesh<sofa::geometry::QuadraticPyramid>, 5>
{
    using ElementType = sofa::geometry::QuadraticPyramid;
    friend BaseDrawMesh;

    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::green(),
        sofa::type::RGBAColor::teal(),
        sofa::type::RGBAColor::navy(),
        sofa::type::RGBAColor::gold(),
        sofa::type::RGBAColor::purple()
    };

private:
    template<class PositionContainer, class IndicesContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& colors)
    {
        if (!topology)
            return;

        const auto& elements = topology->getElements<sofa::geometry::QuadraticPyramid>();

        // 1 quad base (subdivided into 4 quads) + 4 triangular faces (each subdivided into 4 triangles)
        constexpr std::size_t nbSubQuads = 4;
        constexpr std::size_t nbSubTriangles = 4;

        renderedPoints[0].resize(elementIndices.size() * nbSubQuads * 4);
        renderedPoints[1].resize(elementIndices.size() * nbSubTriangles * 3);
        renderedPoints[2].resize(elementIndices.size() * nbSubTriangles * 3);
        renderedPoints[3].resize(elementIndices.size() * nbSubTriangles * 3);
        renderedPoints[4].resize(elementIndices.size() * nbSubTriangles * 3);

        // Base quad (0,1,2,3) subdivided with center node 13
        static constexpr std::array<std::array<std::size_t, 4>, 4> baseQuadSubdivision {{
            {{0, 5, 13, 8}}, {{5, 1, 6, 13}}, {{13, 6, 2, 7}}, {{8, 13, 7, 3}}
        }};

        // Triangular faces subdivided
        static constexpr std::array<std::array<std::array<std::size_t, 3>, 4>, 4> triangleFaces {{
            {{{0, 5, 9}, {5, 1, 10}, {9, 10, 4}, {5, 10, 9}}},  // face 0-1-4
            {{{1, 6, 10}, {6, 2, 11}, {10, 11, 4}, {6, 11, 10}}},  // face 1-2-4
            {{{2, 7, 11}, {7, 3, 12}, {11, 12, 4}, {7, 12, 11}}},  // face 2-3-4
            {{{3, 8, 12}, {8, 0, 9}, {12, 9, 4}, {8, 9, 12}}}   // face 3-0-4
        }};

        std::array<std::size_t, 5> pointId{};

        for (auto idx : elementIndices)
        {
            const auto& element = elements[idx];
            const auto center = this->elementCenter(position,
                sofa::topology::Pyramid{element[0], element[1], element[2], element[3], element[4]});

            // Draw base quad
            for (const auto& quad : baseQuadSubdivision)
            {
                for (std::size_t v = 0; v < 4; ++v)
                {
                    const auto p = this->applyElementSpace(position[element[quad[v]]], center);
                    renderedPoints[0][pointId[0]++] = sofa::type::toVec3(p);
                }
            }

            // Draw 4 triangular faces
            for (std::size_t faceIdx = 0; faceIdx < 4; ++faceIdx)
            {
                for (const auto& tri : triangleFaces[faceIdx])
                {
                    for (std::size_t v = 0; v < 3; ++v)
                    {
                        const auto p = this->applyElementSpace(position[element[tri[v]]], center);
                        renderedPoints[1 + faceIdx][pointId[1 + faceIdx]++] = sofa::type::toVec3(p);
                    }
                }
            }
        }

        drawTool->drawQuads(renderedPoints[0], colors[0]);
        drawTool->drawTriangles(renderedPoints[1], colors[1]);
        drawTool->drawTriangles(renderedPoints[2], colors[2]);
        drawTool->drawTriangles(renderedPoints[3], colors[3]);
        drawTool->drawTriangles(renderedPoints[4], colors[4]);
    }
};

class SOFA_CORE_API DrawMesh
{
public:

    template<class ElementType, class PositionContainer>
    void drawElements(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const typename DrawElementMesh<ElementType>::ColorContainer& colors = DrawElementMesh<ElementType>::defaultColors)
    {
        std::get<DrawElementMesh<ElementType>>(m_meshes).drawAllElements(drawTool, position, topology, colors);
    }

    void setElementSpace(SReal elementSpace);

    template<class PositionContainer>
    void drawLine(sofa::helper::visual::DrawTool* drawTool, const PositionContainer& position, sofa::core::topology::BaseMeshTopology* topology)
    {
        drawElements<sofa::geometry::Edge>(drawTool, position, topology);
        drawElements<sofa::geometry::QuadraticEdge>(drawTool, position, topology);
    }

    template<class PositionContainer>
    void drawSurface(sofa::helper::visual::DrawTool* drawTool, const PositionContainer& position, sofa::core::topology::BaseMeshTopology* topology)
    {
        drawElements<sofa::geometry::Triangle>(drawTool, position, topology);
        drawElements<sofa::geometry::Quad>(drawTool, position, topology);
        drawElements<sofa::geometry::QuadraticTriangle>(drawTool, position, topology);
        drawElements<sofa::geometry::QuadraticQuad>(drawTool, position, topology);
    }

    template<class PositionContainer>
    void drawVolume(sofa::helper::visual::DrawTool* drawTool, const PositionContainer& position, sofa::core::topology::BaseMeshTopology* topology)
    {
        drawElements<sofa::geometry::Tetrahedron>(drawTool, position, topology);
        drawElements<sofa::geometry::QuadraticTetrahedron>(drawTool, position, topology);
        drawElements<sofa::geometry::Hexahedron>(drawTool, position, topology);
        drawElements<sofa::geometry::QuadraticHexahedron>(drawTool, position, topology);
        drawElements<sofa::geometry::Prism>(drawTool, position, topology);
        drawElements<sofa::geometry::QuadraticPrism>(drawTool, position, topology);
        drawElements<sofa::geometry::Pyramid>(drawTool, position, topology);
        drawElements<sofa::geometry::QuadraticPyramid>(drawTool, position, topology);
    }

    template<class PositionContainer>
    void draw(sofa::helper::visual::DrawTool* drawTool, const PositionContainer& position, sofa::core::topology::BaseMeshTopology* topology)
    {
        if (!topology)
        {
            return;
        }

        const auto hasTriangles = !topology->getTriangles().empty();
        const auto hasQTriangle = !topology->getElements<sofa::geometry::QuadraticTriangle>().empty();
        const auto hasQuads = !topology->getQuads().empty();
        const auto hasQQuad = !topology->getElements<sofa::geometry::QuadraticQuad>().empty();

        const auto hasSurfaceElements = hasTriangles || hasQTriangle || hasQuads || hasQQuad;

        const auto hasTetra = !topology->getTetrahedra().empty();
        const auto hasQTetra = !topology->getElements<sofa::geometry::QuadraticTetrahedron>().empty();
        const auto hasHexa = !topology->getHexahedra().empty();
        const auto hasQHexa = !topology->getElements<sofa::geometry::QuadraticHexahedron>().empty();
        const auto hasPrism = !topology->getPrisms().empty();
        const auto hasQPrism = !topology->getElements<sofa::geometry::QuadraticPrism>().empty();
        const auto hasPyramid = !topology->getPyramids().empty();
        const auto hasQPyramid = !topology->getElements<sofa::geometry::QuadraticPyramid>().empty();

        const bool hasVolumeElements = hasTetra || hasQTetra || hasHexa || hasQHexa || hasPrism || hasQPrism || hasPyramid || hasQPyramid;

        if (!hasSurfaceElements && !hasVolumeElements)
        {
            drawLine(drawTool, position, topology);
        }
        else
        {
            if (!hasVolumeElements)
            {
                drawSurface(drawTool, position, topology);
            }
            else
            {
                drawVolume(drawTool, position, topology);
            }
        }
    }

private:
    std::tuple<
        DrawElementMesh<sofa::geometry::Edge>,
        DrawElementMesh<sofa::geometry::QuadraticEdge>,
        DrawElementMesh<sofa::geometry::Triangle>,
        DrawElementMesh<sofa::geometry::QuadraticTriangle>,
        DrawElementMesh<sofa::geometry::Quad>,
        DrawElementMesh<sofa::geometry::QuadraticQuad>,
        DrawElementMesh<sofa::geometry::Tetrahedron>,
        DrawElementMesh<sofa::geometry::QuadraticTetrahedron>,
        DrawElementMesh<sofa::geometry::Hexahedron>,
        DrawElementMesh<sofa::geometry::QuadraticHexahedron>,
        DrawElementMesh<sofa::geometry::Prism>,
        DrawElementMesh<sofa::geometry::QuadraticPrism>,
        DrawElementMesh<sofa::geometry::Pyramid>,
        DrawElementMesh<sofa::geometry::QuadraticPyramid>
    > m_meshes;
};

}  // namespace sofa::core::visual
