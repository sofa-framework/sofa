/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_OGLGRID_H
#define SOFA_OGLGRID_H
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/RGBAColor.h>

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

    Data<std::string> plane; ///< Plane of the grid
    PLANE internalPlane;

    Data<float> size; ///< Size of the squared grid
    Data<int> nbSubdiv; ///< Number of subdivisions

    Data<defaulttype::RGBAColor> color; ///< Color of the lines in the grid. default=(0.34,0.34,0.34,1.0)
    Data<float> thickness; ///< Thickness of the lines in the grid
    Data<bool> draw; ///< Display the grid or not

    OglGrid():
        plane(initData(&plane, std::string("z"),  "plane", "Plane of the grid")),
        size(initData(&size, 10.0f,  "size", "Size of the squared grid")),
        nbSubdiv(initData(&nbSubdiv, 16,  "nbSubdiv", "Number of subdivisions")),
        color(initData(&color, defaulttype::RGBAColor(0.34117647058f,0.34117647058f,0.34117647058f,1.0f),  "color", "Color of the lines in the grid. default=(0.34,0.34,0.34,1.0)")),
        thickness(initData(&thickness, 1.0f,  "thickness", "Thickness of the lines in the grid")),
        draw(initData(&draw, true,  "draw", "Display the grid or not"))
    {}

    virtual void init() override;
    virtual void reinit() override;
    virtual void drawVisual(const core::visual::VisualParams*) override;
    virtual void updateVisual() override;

protected:

};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif //SOFA_OGLGRID_H
