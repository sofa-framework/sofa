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

namespace sofa::core::visual
{

template<class Derived>
struct BaseDrawColoredMesh
{
    SReal elementSpace { 0.125_sreal };

    virtual ~BaseDrawColoredMesh() = default;

    template<class PositionContainer, class ColorContainer>
    void drawAllElements(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const ColorContainer& nodesColors)
    {
        const auto totalNbElements = topology->getNbElements<typename Derived::ElementType>();
        const auto elementsToDraw = sofa::helper::IotaView(static_cast<decltype(totalNbElements)>(0), totalNbElements);
        drawSomeElements(drawTool, position, topology, elementsToDraw, nodesColors);
    }

    template<class PositionContainer, class IndicesContainer, class ColorContainer>
    void drawSomeElements(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& nodesColors)
    {
        if (!drawTool)
            return;
        if (!topology)
            return;

        static_cast<Derived&>(*this).doDraw(drawTool, position, topology, elementIndices, nodesColors);
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

    std::vector<sofa::type::Vec3> renderedPoints;
    std::vector<sofa::type::RGBAColor> colors;
};

template<class ElementType>
struct DrawElementColoredMesh{};

template<>
struct SOFA_CORE_API DrawElementColoredMesh<sofa::geometry::Tetrahedron>
    : public BaseDrawColoredMesh<DrawElementColoredMesh<sofa::geometry::Tetrahedron>>
{
    using ElementType = sofa::geometry::Tetrahedron;
    friend BaseDrawColoredMesh;
    static constexpr std::size_t NumberTrianglesInTetrahedron = 4;

    template<class PositionContainer, class IndicesContainer, class ColorContainer>
    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const PositionContainer& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const IndicesContainer& elementIndices,
        const ColorContainer& nodesColors)
    {
        const auto& elements = topology->getTetrahedra();
        const auto& facets = topology->getTriangles();

        if(facets.empty())
        {
            static bool firstTime = true;
            if (firstTime)
            {
                msg_error("DrawElementColoredMesh<Tetrahedron>") << "Drawing tetrahedra needs the associated triangles in the topology.";
                firstTime = false;
            }
            return;
        }

        renderedPoints.clear();
        colors.clear();

        for (auto i : elementIndices)
        {
            const auto& element = elements[i];
            const auto& facetsInElement = topology->getTrianglesInTetrahedron(i);
            const auto center = this->elementCenter(position, element);

            for (std::size_t j = 0; j < NumberTrianglesInTetrahedron; ++j)
            {
                const auto faceId = facetsInElement[j];
                for (const auto vertexId : facets[faceId])
                {
                    const auto p = this->applyElementSpace(position[vertexId], center);

                    renderedPoints.push_back(p);
                    colors.push_back(nodesColors[vertexId]);
                }
            }
        }

        drawTool->drawTriangles(renderedPoints, colors);
    }

};

class SOFA_CORE_API DrawColoredMesh
{
public:

    void setElementSpace(SReal elementSpace);

    template<class PositionContainer, class ColorContainer>
    void draw(sofa::helper::visual::DrawTool* drawTool, const PositionContainer& position, const ColorContainer& nodesColors, sofa::core::topology::BaseMeshTopology* topology)
    {
        std::get<DrawElementColoredMesh<sofa::geometry::Tetrahedron>>(m_meshes).drawAllElements(drawTool, position, topology, nodesColors);
    }

private:
    std::tuple<
        // DrawElementColoredMesh<sofa::geometry::Edge>,
        // DrawElementColoredMesh<sofa::geometry::Triangle>,
        // DrawElementColoredMesh<sofa::geometry::Quad>,
        DrawElementColoredMesh<sofa::geometry::Tetrahedron>
        // DrawElementColoredMesh<sofa::geometry::Hexahedron>,
        // DrawElementColoredMesh<sofa::geometry::Prism>,
        // DrawElementColoredMesh<sofa::geometry::Pyramid>
    > m_meshes;
};

}
