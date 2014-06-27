/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_OGLLINEAXIS_H
#define SOFA_OGLLINEAXIS_H

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class OglLineAxis : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglLineAxis, VisualModel);

    Data<std::string> axis;
    Data<float> size;
    Data<float> thickness;
    Data<bool> draw;

    OglLineAxis():
        axis(initData(&axis, std::string("xyz"),  "axis", "Axis to draw")),
        size(initData(&size, (float)(10.0),  "size", "Size of the squared grid")),
        thickness(initData(&thickness, (float)(1.0),  "thickness", "Thickness of the lines in the grid")),
        draw(initData(&draw, true,  "draw", "Display the grid or not")),
        drawX(true), drawY(true), drawZ(true)
    {}

    virtual void init();
    virtual void reinit();
    virtual void drawVisual(const core::visual::VisualParams*);
    virtual void updateVisual();

protected:

    bool drawX;
    bool drawY;
    bool drawZ;

};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif //SOFA_OGLLINEAXIS_H
