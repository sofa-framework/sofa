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

#include <sofa/component/visual/VisualGrid.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::visual
{

int VisualGridClass = core::RegisterObject("Display a simple grid")
        .add< VisualGrid>()
        ;

using namespace sofa::defaulttype;

VisualGrid::VisualGrid()
    : d_plane(initData(&d_plane, std::string("z"),  "plane", "Plane of the grid"))
    , d_size(initData(&d_size, 10.0f,  "size", "Size of the squared grid"))
    , d_nbSubdiv(initData(&d_nbSubdiv, 16,  "nbSubdiv", "Number of subdivisions"))
    , d_color(initData(&d_color, sofa::type::RGBAColor(0.34117647058f,0.34117647058f,0.34117647058f,1.0f),  "color", "Color of the lines in the grid. default=(0.34,0.34,0.34,1.0)"))
    , d_thickness(initData(&d_thickness, 1.0f,  "thickness", "Thickness of the lines in the grid"))
    , internalPlane(PLANE_Z)
{
    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Loading);
    addUpdateCallback("buildGrid", {&d_plane, &d_size, &d_nbSubdiv}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        updateVisual();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {});
}

void VisualGrid::init()
{
    Inherit1::init();
    updateVisual();

    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void VisualGrid::reinit()
{
    updateVisual();
}

void VisualGrid::updateVisual()
{
    const auto planeValue = d_plane.getValue();

    if (planeValue == "x" ||
             planeValue == "X" ||
            planeValue == "zOy" ||
            planeValue == "ZOY" ||
            planeValue == "yOz" ||
            planeValue == "YOZ")
    {
        internalPlane = PLANE_X;
    }
    else if (planeValue == "y" ||
             planeValue == "Y" ||
             planeValue == "zOx" ||
             planeValue == "ZOX" ||
             planeValue == "xOz" ||
             planeValue == "XOZ")
    {
        internalPlane = PLANE_Y;
    }
    else if (planeValue == "z" ||
             planeValue == "Z" ||
             planeValue == "xOy" ||
             planeValue == "XOY" ||
             planeValue == "yOx" ||
             planeValue == "YOX")
    {
        internalPlane = PLANE_Z;
    }
    else
    {
        msg_error() << "Plane parameter " << d_plane.getValue() << " not recognized. Set to z instead";
        d_plane.setValue("z");
        internalPlane = PLANE_Z;
    }

    const int nb = d_nbSubdiv.getValue();
    if (nb < 2)
    {
        msg_error() << "The Data " << d_nbSubdiv.getName() << " should be >= 2";
        d_nbSubdiv.setValue(2);
    }

    buildGrid();

    //bounding box for the camera
    auto s = d_size.getValue() * 0.5f;
    sofa::type::Vec3f min(-s, -s, -s);
    sofa::type::Vec3f max( s,  s,  s);
    min[internalPlane] = -s * 0.2f;
    max[internalPlane] =  s * 0.2f;
    f_bbox.setValue(sofa::type::BoundingBox(min,max));
}


void VisualGrid::buildGrid()
{
    m_drawnPoints.clear();

    const unsigned int nb = d_nbSubdiv.getValue();
    const float s = d_size.getValue();
    const float hs = s / 2; //half the size
    const float s_nb = s / static_cast<float>(nb); //space between points

    m_drawnPoints.reserve(4u * (nb + 1));

    switch(internalPlane)
    {
    case PLANE_X:
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            m_drawnPoints.emplace_back(0.0, -hs + i * s_nb, -hs);
            m_drawnPoints.emplace_back(0.0, -hs + i * s_nb,  hs);
        }
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            m_drawnPoints.emplace_back(0.0, -hs, -hs + i * s_nb);
            m_drawnPoints.emplace_back(0.0,  hs, -hs + i * s_nb);
        }
        break;
    case PLANE_Y:
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            m_drawnPoints.emplace_back(-hs, 0.0, -hs + i * s_nb);
            m_drawnPoints.emplace_back( hs, 0.0, -hs + i * s_nb);
        }
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            m_drawnPoints.emplace_back(-hs + i * s_nb, 0.0, -hs);
            m_drawnPoints.emplace_back(-hs + i * s_nb, 0.0,  hs);
        }
        break;
    case PLANE_Z:
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            m_drawnPoints.emplace_back(-hs, -hs + i * s_nb, 0.0);
            m_drawnPoints.emplace_back( hs, -hs + i * s_nb, 0.0);
        }
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            m_drawnPoints.emplace_back(-hs + i * s_nb, -hs, 0.0);
            m_drawnPoints.emplace_back(-hs + i * s_nb,  hs, 0.0);
        }
        break;
    }
}

void VisualGrid::doDrawVisual(const core::visual::VisualParams* vparams)
{
    vparams->drawTool()->disableLighting();
    vparams->drawTool()->drawLines(m_drawnPoints, d_thickness.getValue(), d_color.getValue());

}

} // namespace sofa::component::visual
