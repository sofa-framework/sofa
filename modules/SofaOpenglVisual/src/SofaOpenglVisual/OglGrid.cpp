/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaOpenglVisual/OglGrid.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

static int OglGridClass = core::RegisterObject("Display a simple grid")
        .add< component::visualmodel::OglGrid>()
        ;

using namespace sofa::defaulttype;

OglGrid::OglGrid()
    : d_plane(initData(&d_plane, std::string("z"),  "plane", "Plane of the grid"))
    , d_size(initData(&d_size, 10.0f,  "size", "Size of the squared grid"))
    , d_nbSubdiv(initData(&d_nbSubdiv, 16,  "nbSubdiv", "Number of subdivisions"))
    , d_color(initData(&d_color, defaulttype::RGBAColor(0.34117647058f,0.34117647058f,0.34117647058f,1.0f),  "color", "Color of the lines in the grid. default=(0.34,0.34,0.34,1.0)"))
    , d_thickness(initData(&d_thickness, 1.0f,  "thickness", "Thickness of the lines in the grid"))
    , d_draw(initData(&d_draw, true,  "draw", "Display the grid or not"))
{}

void OglGrid::init()
{
    updateVisual();
}

void OglGrid::reinit()
{
    updateVisual();
}

void OglGrid::updateVisual()
{
    if (d_plane.getValue() == "x" ||
             d_plane.getValue() == "X" ||
            d_plane.getValue() == "zOy" ||
            d_plane.getValue() == "ZOY" ||
            d_plane.getValue() == "yOz" ||
            d_plane.getValue() == "YOZ")
    {
        internalPlane = PLANE_X;
    }
    else if (d_plane.getValue() == "y" ||
             d_plane.getValue() == "Y" ||
             d_plane.getValue() == "zOx" ||
             d_plane.getValue() == "ZOX" ||
             d_plane.getValue() == "xOz" ||
             d_plane.getValue() == "XOZ")
    {
        internalPlane = PLANE_Y;
    }
    else if (d_plane.getValue() == "z" ||
             d_plane.getValue() == "Z" ||
             d_plane.getValue() == "xOy" ||
             d_plane.getValue() == "XOY" ||
             d_plane.getValue() == "yOx" ||
             d_plane.getValue() == "YOX")
    {
        internalPlane = PLANE_Z;
    }
    else
    {
        serr << "Plane parameter " << d_plane.getValue() << " not recognized. Set to z instead" << sendl;
        d_plane.setValue("z");
        internalPlane = PLANE_Z;
    }

    int nb = d_nbSubdiv.getValue();
    if (nb < 2)
    {
        serr << "nbSubdiv should be > 2" << sendl;
        d_nbSubdiv.setValue(2);
    }
}


void OglGrid::drawVisual(const core::visual::VisualParams* vparams)
{
    if (!d_draw.getValue()) return;

    std::vector<Vector3> points;

    unsigned int nb = d_nbSubdiv.getValue();
    float s = d_size.getValue();

    switch(internalPlane)
    {
    case PLANE_X:
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            points.push_back(Vector3(0.0, -s*0.5 + i * s / nb, -s*0.5));
            points.push_back(Vector3(0.0, -s*0.5 + i * s / nb,  s*0.5));
        }
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            points.push_back(Vector3(0.0, -s*0.5, -s*0.5 + i * s / nb));
            points.push_back(Vector3(0.0,  s*0.5, -s*0.5 + i * s / nb));
        }
        break;
    case PLANE_Y:
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            points.push_back(Vector3(-s*0.5, 0.0, -s*0.5 + i * s / nb));
            points.push_back(Vector3( s*0.5, 0.0, -s*0.5 + i * s / nb));
        }
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            points.push_back(Vector3(-s*0.5 + i * s / nb, 0.0, -s*0.5));
            points.push_back(Vector3(-s*0.5 + i * s / nb, 0.0,  s*0.5));
        }
        break;
    case PLANE_Z:
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            points.push_back(Vector3(-s*0.5, -s*0.5 + i * s / nb, 0.0));
            points.push_back(Vector3( s*0.5, -s*0.5 + i * s / nb, 0.0));
        }
        for (unsigned int i = 0 ; i < nb+1; ++i)
        {
            points.push_back(Vector3(-s*0.5 + i * s / nb, -s*0.5, 0.0));
            points.push_back(Vector3(-s*0.5 + i * s / nb,  s*0.5, 0.0));
        }
        break;
    }

    vparams->drawTool()->drawLines(points, d_thickness.getValue(), d_color.getValue());
}

} // namespace visualmodel

} // namespace component

} // namespace sofa
