/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

SOFA_DECL_CLASS(OglGrid)

int OglGridClass = core::RegisterObject("Display a simple grid")
        .add< component::visualmodel::OglGrid>()
        ;

using namespace sofa::defaulttype;


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
    if (plane.getValue() == "x" ||
             plane.getValue() == "X" ||
            plane.getValue() == "zOy" ||
            plane.getValue() == "ZOY" ||
            plane.getValue() == "yOz" ||
            plane.getValue() == "YOZ")
    {
        internalPlane = PLANE_X;
    }
    else if (plane.getValue() == "y" ||
             plane.getValue() == "Y" ||
             plane.getValue() == "zOx" ||
             plane.getValue() == "ZOX" ||
             plane.getValue() == "xOz" ||
             plane.getValue() == "XOZ")
    {
        internalPlane = PLANE_Y;
    }
    else if (plane.getValue() == "z" ||
             plane.getValue() == "Z" ||
             plane.getValue() == "xOy" ||
             plane.getValue() == "XOY" ||
             plane.getValue() == "yOx" ||
             plane.getValue() == "YOX")
    {
        internalPlane = PLANE_Z;
    }
    else
    {
        serr << "Plane parameter " << plane.getValue() << " not recognized. Set to z instead" << sendl;
        plane.setValue("z");
        internalPlane = PLANE_Z;
    }

    int nb = nbSubdiv.getValue();
    if (nb < 2)
    {
        serr << "nbSubdiv should be > 2" << sendl;
        nbSubdiv.setValue(2);
    }

    //bounding box for the camera
//    Real s = size.getValue();
//    Coord min,max;
//    switch(internalPlane)
//    {
//        case PLANE_X:
//            min = Coord(-s*0.1, -s*0.5, -s*0.5);
//            max = Coord(s*0.1, s*0.5, s*0.5);
//            break;
//        case PLANE_Y:
//            min = Coord(-s*0.5, -s*0.1, -s*0.5);
//            max = Coord(s*0.5, s*0.1, s*0.5);
//            break;
//        case PLANE_Z:
//            min = Coord(-s*0.5, -s*0.5, -s*0.1);
//            max = Coord(s*0.5, s*0.5, s*0.1);
//            break;
//    }
//    f_bbox.setValue(sofa::defaulttype::BoundingBox(min,max));

}


void OglGrid::drawVisual(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!draw.getValue()) return;

    std::vector<Vector3> points;

    unsigned int nb = nbSubdiv.getValue();
    float s = size.getValue();

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

    vparams->drawTool()->drawLines(points, thickness.getValue(), color.getValue());

#endif
}

} // namespace visualmodel

} // namespace component

} // namespace sofa
