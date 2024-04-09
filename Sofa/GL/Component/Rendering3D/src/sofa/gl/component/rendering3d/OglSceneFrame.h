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
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa::gl::component::rendering3d
{

class SOFA_GL_COMPONENT_RENDERING3D_API OglSceneFrame : public core::visual::VisualModel
{

public:
    SOFA_CLASS(OglSceneFrame, VisualModel);

    typedef core::visual::VisualParams::Viewport Viewport;

    Data<bool> d_drawFrame; ///< Display the frame or not
    Data<sofa::helper::OptionsGroup> d_style; ///< Style of the frame
    Data<sofa::helper::OptionsGroup> d_alignment; ///< Alignment of the frame in the view
    Data<int> d_viewportSize; ///< Size of the viewport where the frame is rendered

    OglSceneFrame();

    void init() override;
    void reinit() override;
    void draw(const core::visual::VisualParams*) override;

private:
    static void drawArrows(const core::visual::VisualParams* vparams);
    static void drawCylinders(const core::visual::VisualParams* vparams);
    static void drawCubeCones(const core::visual::VisualParams* vparams);
};

} // namespace sofa::gl::component::rendering3d
