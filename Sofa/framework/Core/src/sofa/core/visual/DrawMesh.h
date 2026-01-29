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
        const auto totalNbElements = topology->getNbElements<typename Derived::ElementType>();
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

        const auto stateLifeCycle = drawTool->makeStateLifeCycle();
        drawTool->disableLighting();

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

        for ( auto& p : renderedPoints)
        {
            p.resize(elementIndices.size() * sofa::geometry::Quad::NumberOfNodes);
        }

        std::size_t renderedPointId {};
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
    }

    template<class PositionContainer>
    void drawSurface(sofa::helper::visual::DrawTool* drawTool, const PositionContainer& position, sofa::core::topology::BaseMeshTopology* topology)
    {
        drawElements<sofa::geometry::Triangle>(drawTool, position, topology);
        drawElements<sofa::geometry::Quad>(drawTool, position, topology);
    }

    template<class PositionContainer>
    void drawVolume(sofa::helper::visual::DrawTool* drawTool, const PositionContainer& position, sofa::core::topology::BaseMeshTopology* topology)
    {
        drawElements<sofa::geometry::Tetrahedron>(drawTool, position, topology);
        drawElements<sofa::geometry::Hexahedron>(drawTool, position, topology);
    }

    template<class PositionContainer>
    void draw(sofa::helper::visual::DrawTool* drawTool, const PositionContainer& position, sofa::core::topology::BaseMeshTopology* topology)
    {
        if (!topology)
        {
            return;
        }

        const auto hasTriangles = !topology->getTriangles().empty();
        const auto hasQuads = !topology->getQuads().empty();

        const auto hasSurfaceElements = hasTriangles || hasQuads;

        const auto hasTetra = !topology->getTetrahedra().empty();
        const auto hasHexa = !topology->getHexahedra().empty();

        const bool hasVolumeElements = hasTetra || hasHexa;

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
        DrawElementMesh<sofa::geometry::Triangle>,
        DrawElementMesh<sofa::geometry::Quad>,
        DrawElementMesh<sofa::geometry::Tetrahedron>,
        DrawElementMesh<sofa::geometry::Hexahedron>
    > m_meshes;
};

}  // namespace sofa::core::visual
