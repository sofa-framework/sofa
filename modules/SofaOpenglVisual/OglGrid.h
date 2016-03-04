/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_OGLGRID_H
#define SOFA_OGLGRID_H
#include "config.h"

#include <sofa/core/visual/VisualModel.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class OglGrid : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglGrid, VisualModel);

    typedef sofa::defaulttype::Vector3 Vector3;

    enum PLANE {PLANE_X, PLANE_Y, PLANE_Z};

    Data<std::string> plane;
    PLANE internalPlane;

    Data<float> size;
    Data<int> nbSubdiv;

    Data<sofa::defaulttype::Vec<4, float> > color;
    Data<float> thickness;
    Data<bool> draw;

    OglGrid():
        plane(initData(&plane, std::string("z"),  "plane", "Plane of the grid")),
        size(initData(&size, 10.0f,  "size", "Size of the squared grid")),
        nbSubdiv(initData(&nbSubdiv, 16,  "nbSubdiv", "Number of subdivisions")),
        color(initData(&color, sofa::defaulttype::Vec<4, float>(0.34117647058f,0.34117647058f,0.34117647058f,1.0f),  "color", "Color of the lines in the grid")),
        thickness(initData(&thickness, 1.0f,  "thickness", "Thickness of the lines in the grid")),
        draw(initData(&draw, true,  "draw", "Display the grid or not"))
    {}

    virtual void init();
    virtual void reinit();
    virtual void drawVisual(const core::visual::VisualParams*);
    virtual void updateVisual();

protected:

};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif //SOFA_OGLGRID_H
