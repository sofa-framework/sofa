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
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/Topology.h>

#include <sofa/component/solidmechanics/fem/hyperelastic/material/BoyceAndArruda.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/NeoHookean.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/MooneyRivlin.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/VerondaWestman.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/STVenantKirchhoff.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/Costa.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/Ogden.h>


namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
void selectColors(const std::string& materialName, sofa::type::RGBAColor& color1, sofa::type::RGBAColor& color2, sofa::type::RGBAColor& color3, sofa::type::RGBAColor& color4)
{
    if (materialName == material::BoyceAndArruda<DataTypes>::Name)
    {
        color1 = type::RGBAColor(0.0, 1.0, 0.0, 1.0);
        color2 = type::RGBAColor(0.5, 1.0, 0.0, 1.0);
        color3 = type::RGBAColor(1.0, 1.0, 0.0, 1.0);
        color4 = type::RGBAColor(1.0, 1.0, 0.5, 1.0);
    }
    else if (materialName == material::STVenantKirchhoff<DataTypes>::Name)
    {
        color1 = type::RGBAColor(1.0,0.0,0.0,1.0);
        color2 = type::RGBAColor(1.0,0.0,0.5,1.0);
        color3 = type::RGBAColor(1.0,1.0,0.0,1.0);
        color4 = type::RGBAColor(1.0,0.5,1.0,1.0);
    }
    else if (materialName == material::NeoHookean<DataTypes>::Name)
    {
        color1 = type::RGBAColor(0.0, 1.0, 1.0, 1.0);
        color2 = type::RGBAColor(0.5, 0.0, 1.0, 1.0);
        color3 = type::RGBAColor(1.0, 0.0, 1.0, 1.0);
        color4 = type::RGBAColor(1.0, 0.5, 1.0, 1.0);
    }
    else if (materialName == material::MooneyRivlin<DataTypes>::Name)
    {
        color1 = type::RGBAColor(0.0, 1.0, 0.0, 1.0);
        color2 = type::RGBAColor(0.0, 1.0, 0.5, 1.0);
        color3 = type::RGBAColor(0.0, 1.0, 1.0, 1.0);
        color4 = type::RGBAColor(0.5, 1.0, 1.0, 1.0);
    }
    else if (materialName == material::VerondaWestman<DataTypes>::Name)
    {
        color1 = type::RGBAColor(0.0, 1.0, 0.0, 1.0);
        color2 = type::RGBAColor(0.5, 1.0, 0.0, 1.0);
        color3 = type::RGBAColor(1.0, 1.0, 0.0, 1.0);
        color4 = type::RGBAColor(1.0, 1.0, 0.5, 1.0);
    }
    else if (materialName == material::Costa<DataTypes>::Name)
    {
        color1 = type::RGBAColor(0.0, 1.0, 0.0, 1.0);
        color2 = type::RGBAColor(0.5, 1.0, 0.0, 1.0);
        color3 = type::RGBAColor(1.0, 1.0, 0.0, 1.0);
        color4 = type::RGBAColor(1.0, 1.0, 0.5, 1.0);
    }
    else if (materialName == material::Ogden<DataTypes>::Name)
    {
        color1 = type::RGBAColor(0.0, 1.0, 0.0, 1.0);
        color2 = type::RGBAColor(0.5, 1.0, 0.0, 1.0);
        color3 = type::RGBAColor(1.0, 1.0, 0.0, 1.0);
        color4 = type::RGBAColor(1.0, 1.0, 0.5, 1.0);
    }
    else
    {
        color1 = type::RGBAColor(0.0, 1.0, 0.0, 1.0);
        color2 = type::RGBAColor(0.5, 1.0, 0.0, 1.0);
        color3 = type::RGBAColor(1.0, 1.0, 0.0, 1.0);
        color4 = type::RGBAColor(1.0, 1.0, 0.5, 1.0);
    }
}

template <class DataTypes>
void drawHyperelasticTets(const core::visual::VisualParams* vparams,
                          const typename DataTypes::VecCoord& x,
                          core::topology::BaseMeshTopology* topology,
                          const std::string& materialName,
                          const sofa::type::vector<core::topology::Topology::TetrahedronID>& indicesToDraw)
{
    std::vector<type::Vec3 > points[4];
    for(auto& p : points)
    {
        p.reserve(3 * indicesToDraw.size());
    }

    const auto& tetrahedra = topology->getTetrahedra();

    for(const auto i : indicesToDraw)
    {
        const auto t = tetrahedra[i];

        const Index a = t[0];
        const Index b = t[1];
        const Index c = t[2];
        const Index d = t[3];

        const auto center = (x[a] + x[b] + x[c] + x[d]) * 0.125;
        const auto pa = (x[a] + center) * 0.666667;
        const auto pb = (x[b] + center) * 0.666667;
        const auto pc = (x[c] + center) * 0.666667;
        const auto pd = (x[d] + center) * 0.666667;

        points[0].push_back(pa);
        points[0].push_back(pb);
        points[0].push_back(pc);

        points[1].push_back(pb);
        points[1].push_back(pc);
        points[1].push_back(pd);

        points[2].push_back(pc);
        points[2].push_back(pd);
        points[2].push_back(pa);

        points[3].push_back(pd);
        points[3].push_back(pa);
        points[3].push_back(pb);
    }

    std::array<sofa::type::RGBAColor, 4> colors;
    selectColors<DataTypes>(materialName, colors[0], colors[1], colors[2], colors[3]);

    vparams->drawTool()->drawTriangles(points[0], colors[0]);
    vparams->drawTool()->drawTriangles(points[1], colors[1]);
    vparams->drawTool()->drawTriangles(points[2], colors[2]);
    vparams->drawTool()->drawTriangles(points[3], colors[3]);
}

template <class DataTypes>
void drawHyperelasticTets(const core::visual::VisualParams* vparams,
                          const typename DataTypes::VecCoord& x,
                          core::topology::BaseMeshTopology* topology,
                          const std::string& materialName)
{
    sofa::type::vector<core::topology::Topology::TetrahedronID> allIndices(topology->getNbTetrahedra());
    std::iota(allIndices.begin(), allIndices.end(), static_cast<core::topology::Topology::TetrahedronID>(0));
    drawHyperelasticTets<DataTypes>(vparams, x, topology, materialName, allIndices);
}


}
