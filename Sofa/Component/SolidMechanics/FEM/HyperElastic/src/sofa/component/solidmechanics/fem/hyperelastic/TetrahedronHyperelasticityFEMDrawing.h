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
#include <sofa/component/solidmechanics/fem/hyperelastic/material/BoyceAndArruda.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/Costa.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/MooneyRivlin.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/NeoHookean.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/Ogden.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/STVenantKirchhoff.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/material/VerondaWestman.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/visual/DrawMesh.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
void selectColors(const std::string& materialName, sofa::type::RGBAColor& color1, sofa::type::RGBAColor& color2, sofa::type::RGBAColor& color3, sofa::type::RGBAColor& color4)
{
    if (materialName == material::BoyceAndArruda<DataTypes>::Name)
    {
        color1 = type::RGBAColor::green();
        color2 = type::RGBAColor::lime();
        color3 = type::RGBAColor::yellow();
        color4 = type::RGBAColor(1.0, 1.0, 0.5, 1.0);
    }
    else if (materialName == material::STVenantKirchhoff<DataTypes>::Name)
    {
        color1 = type::RGBAColor::red();
        color2 = type::RGBAColor::pink();
        color3 = type::RGBAColor::yellow();
        color4 = type::RGBAColor(1.0,0.5,1.0,1.0);
    }
    else if (materialName == material::NeoHookean<DataTypes>::Name)
    {
        color1 = type::RGBAColor::cyan();
        color2 = type::RGBAColor(0.5, 0.0, 1.0, 1.0);
        color3 = type::RGBAColor::magenta();
        color4 = type::RGBAColor(1.0, 0.5, 1.0, 1.0);
    }
    else if (materialName == material::MooneyRivlin<DataTypes>::Name)
    {
        color1 = type::RGBAColor::green();
        color2 = type::RGBAColor(0.0, 1.0, 0.5, 1.0);
        color3 = type::RGBAColor::cyan();
        color4 = type::RGBAColor(0.5, 1.0, 1.0, 1.0);
    }
    else if (materialName == material::VerondaWestman<DataTypes>::Name)
    {
        color1 = type::RGBAColor::green();
        color2 = type::RGBAColor(0.5, 1.0, 0.0, 1.0);
        color3 = type::RGBAColor(1.0, 1.0, 0.0, 1.0);
        color4 = type::RGBAColor(1.0, 1.0, 0.5, 1.0);
    }
    else if (materialName == material::Costa<DataTypes>::Name)
    {
        color1 = type::RGBAColor::green();
        color2 = type::RGBAColor(0.5, 1.0, 0.0, 1.0);
        color3 = type::RGBAColor(1.0, 1.0, 0.0, 1.0);
        color4 = type::RGBAColor(1.0, 1.0, 0.5, 1.0);
    }
    else if (materialName == material::Ogden<DataTypes>::Name)
    {
        color1 = type::RGBAColor::green();
        color2 = type::RGBAColor(0.5, 1.0, 0.0, 1.0);
        color3 = type::RGBAColor(1.0, 1.0, 0.0, 1.0);
        color4 = type::RGBAColor(1.0, 1.0, 0.5, 1.0);
    }
    else
    {
        color1 = type::RGBAColor::green();
        color2 = type::RGBAColor(0.5, 1.0, 0.0, 1.0);
        color3 = type::RGBAColor(1.0, 1.0, 0.0, 1.0);
        color4 = type::RGBAColor(1.0, 1.0, 0.5, 1.0);
    }
}

template <class DataTypes, class IndicesContainer>
void drawHyperelasticTets(const core::visual::VisualParams* vparams,
                          const typename DataTypes::VecCoord& x,
                          core::topology::BaseMeshTopology* topology,
                          const std::string& materialName,
                          const IndicesContainer& indices)
{
    std::array<sofa::type::RGBAColor, 4> colors;
    selectColors<DataTypes>(materialName, colors[0], colors[1], colors[2], colors[3]);

    core::visual::DrawElementMesh<sofa::geometry::Tetrahedron> drawer;
    drawer.drawSomeElements(vparams->drawTool(), x, topology, indices, colors);
}

template <class DataTypes>
void drawHyperelasticTets(const core::visual::VisualParams* vparams,
                          const typename DataTypes::VecCoord& x,
                          core::topology::BaseMeshTopology* topology,
                          const std::string& materialName)
{
    const auto indices = sofa::helper::IotaView(static_cast<sofa::Size>(0), topology->getNbTetrahedra());
    drawHyperelasticTets<DataTypes>(vparams, x, topology, materialName, indices);
}


}
