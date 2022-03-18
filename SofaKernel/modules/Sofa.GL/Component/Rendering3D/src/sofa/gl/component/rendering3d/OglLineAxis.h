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
#include <sofa/gl/component/rendering3d/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::gl::component::rendering3d
{

class OglLineAxis : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglLineAxis, VisualModel);

    Data<std::string> axis; ///< Axis to draw
    Data<float> size; ///< Size of the squared grid
    Data<float> thickness; ///< Thickness of the lines in the grid
    Data<bool> draw; ///< Display the grid or not

    OglLineAxis();

    void init() override;
    void reinit() override;
    void drawVisual(const core::visual::VisualParams*) override;
    void updateVisual() override;

protected:

    bool drawX;
    bool drawY;
    bool drawZ;

};

} // namespace sofa::gl::component::rendering3d
