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

struct SOFA_CORE_API BaseDrawMesh
{
    virtual ~BaseDrawMesh() = default;

    sofa::type::vector< sofa::type::Vec3 > renderedPoints;
    sofa::type::vector< sofa::type::RGBAColor > renderedColors;

    SReal elementSpace { 0.125_sreal };

    virtual void draw(
        sofa::helper::visual::DrawTool* drawTool,
        const type::vector<type::Vec3>& position,
        sofa::core::topology::BaseMeshTopology* topology) = 0;
};

template<class ElementType>
struct DrawElementMesh{};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Triangle> : BaseDrawMesh
{
    void draw(
        sofa::helper::visual::DrawTool* drawTool,
        const type::vector<type::Vec3>& position,
        sofa::core::topology::BaseMeshTopology* topology) override;
};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Tetrahedron> : BaseDrawMesh
{
    void draw(
        sofa::helper::visual::DrawTool* drawTool,
        const type::vector<type::Vec3>& position,
        sofa::core::topology::BaseMeshTopology* topology) override;
};

template<>
struct SOFA_CORE_API DrawElementMesh<sofa::geometry::Hexahedron> : BaseDrawMesh
{
    void draw(
        sofa::helper::visual::DrawTool* drawTool,
        const type::vector<type::Vec3>& position,
        sofa::core::topology::BaseMeshTopology* topology) override;
};

class SOFA_CORE_API DrawMesh
{
public:

    void drawTriangles(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology);
    void drawTetrahedra(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology);
    void drawHexahedra(sofa::helper::visual::DrawTool* drawTool, const type::vector<type::Vec3>& position, sofa::core::topology::BaseMeshTopology* topology);

    void setElementSpace(SReal elementSpace);

   private:

    DrawElementMesh<sofa::geometry::Triangle> m_drawTriangleMesh;
    DrawElementMesh<sofa::geometry::Tetrahedron> m_drawTetrahedronMesh;
    DrawElementMesh<sofa::geometry::Hexahedron> m_drawHexahedronMesh;
};

}  // namespace sofa::core::visual
