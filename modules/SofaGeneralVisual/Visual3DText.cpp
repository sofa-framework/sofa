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

#include <sofa/core/ObjectFactory.h>
#include "Visual3DText.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/RGBAColor.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(Visual3DText)

int Visual3DTextClass = core::RegisterObject("Display 3D camera-oriented text")
        .add< Visual3DText >()
        ;



Visual3DText::Visual3DText()
    : d_text(initData(&d_text, "text", "Test to display"))
    , d_position(initData(&d_position, defaulttype::Vec3f(), "position", "3d position"))
    , d_scale(initData(&d_scale, 1.f, "scale", "text scale"))
    , d_color(initData(&d_color, defaulttype::RGBAColor(1.0,1.0,1.0,1.0), "color", "text color. (default=[1.0,1.0,1.0,1.0])"))
    , d_depthTest(initData(&d_depthTest, true, "depthTest", "perform depth test"))
{
}


void Visual3DText::init()
{
    VisualModel::init();

    reinit();

    updateVisual();
}

void Visual3DText::reinit()
{
}

void Visual3DText::drawTransparent(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if(!vparams->displayFlags().getShowVisualModels()) return;

    const defaulttype::Vec3f& pos = d_position.getValue();
    float scale = d_scale.getValue();

    const bool& depthTest = d_depthTest.getValue();
    if( !depthTest )
    {
        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_DEPTH_TEST);
    }

    vparams->drawTool()->setLightingEnabled(true);


    vparams->drawTool()->draw3DText(pos,scale,d_color.getValue(),d_text.getValue().c_str());


    if( !depthTest )
        glPopAttrib();
#endif /* SOFA_NO_OPENGL */
}


} // namespace visualmodel

} // namespace component

} // namespace sofa

