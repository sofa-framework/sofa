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
#include <sofa/helper/visual/DrawTool.h>

namespace sofa::core::visual
{

template<class Derived>
struct BaseDrawMesh
{
    using ColorContainer = std::array<sofa::type::RGBAColor, Derived::NumberColors>;

    SReal elementSpace { 0.125_sreal };

    void draw(
        sofa::helper::visual::DrawTool* drawTool,
        const type::vector<type::Vec3>& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const ColorContainer& colors = Derived::defaultColors)
    {
        if (!drawTool)
            return;
        if (!topology)
            return;

        const auto stateLifeCycle = drawTool->makeStateLifeCycle();
        drawTool->disableLighting();

        static_cast<Derived&>(*this)->doDraw(drawTool, position, topology, colors);
    }

    sofa::type::Vec3 applyElementSpace(const sofa::type::Vec3& position, const sofa::type::Vec3& elementCenter) const
    {
        return (position - elementCenter) * (1._sreal - elementSpace) + elementCenter;
    }

protected:
    std::array<sofa::type::vector< sofa::type::Vec3 >, Derived::NumberColors> renderedPoints;
};

template<class ElementType>
struct DrawElementMesh{};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Triangle>
    : BaseDrawMesh<DrawElementMesh<sofa::geometry::Triangle>>
{
    static constexpr std::size_t NumberColors = 3;

    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::green(),
        sofa::type::RGBAColor::teal(),
        sofa::type::RGBAColor::blue()
    };

    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const type::vector<type::Vec3>& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const ColorContainer& colors);
};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Tetrahedron>
    : BaseDrawMesh<DrawElementMesh<sofa::geometry::Tetrahedron>>
{
    static constexpr std::size_t NumberTrianglesInTetrahedron = 4;
    static constexpr std::size_t NumberColors = NumberTrianglesInTetrahedron;

    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor::blue(),
        sofa::type::RGBAColor::black(),
        sofa::type::RGBAColor::azure(),
        sofa::type::RGBAColor::cyan()
    };

private:

    void doDraw(sofa::helper::visual::DrawTool* drawTool,
        const type::vector<type::Vec3>& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const ColorContainer& colors);
};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Hexahedron>
    : BaseDrawMesh<DrawElementMesh<sofa::geometry::Hexahedron>>
{
    static constexpr std::size_t NumberQuadsInHexahedron = 6;
    static constexpr std::size_t NumberColors = NumberQuadsInHexahedron;
    static constexpr ColorContainer defaultColors {
        sofa::type::RGBAColor(0.7f,0.7f,0.1f,1.f),
        sofa::type::RGBAColor(0.7f,0.0f,0.0f,1.f),
        sofa::type::RGBAColor(0.0f,0.7f,0.0f,1.f),
        sofa::type::RGBAColor(0.0f,0.0f,0.7f,1.f),
        sofa::type::RGBAColor(0.1f,0.7f,0.7f,1.f),
        sofa::type::RGBAColor(0.7f,0.1f,0.7f,1.f)
    };

    void doDraw(
        sofa::helper::visual::DrawTool* drawTool,
        const type::vector<type::Vec3>& position,
        sofa::core::topology::BaseMeshTopology* topology,
        const ColorContainer& colors);
};

class SOFA_CORE_API DrawMesh
{
public:

    void drawTriangles(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology);
    void drawTetrahedra(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology);
    void drawHexahedra(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology);

    void setElementSpace(SReal elementSpace);

    void drawSurface(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology);
    void drawVolume(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology);

    void draw(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology);

private:

    DrawElementMesh<sofa::geometry::Triangle> m_drawTriangleMesh;
    DrawElementMesh<sofa::geometry::Tetrahedron> m_drawTetrahedronMesh;
    DrawElementMesh<sofa::geometry::Hexahedron> m_drawHexahedronMesh;
};

}  // namespace sofa::core::visual
