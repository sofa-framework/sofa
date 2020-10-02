/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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

#include <sofa/core/ObjectFactory.h>
#include "Visual3DText.h"
#include <sofa/core/visual/VisualParams.h>


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
    , d_color(initData(&d_color, std::string("white"), "color", "text color"))
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
    setColor(d_color.getValue());
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


    vparams->drawTool()->draw3DText(pos,scale,m_color,d_text.getValue().c_str());


    if( !depthTest )
        glPopAttrib();
#endif /* SOFA_NO_OPENGL */
}


void Visual3DText::setColor(float r, float g, float b, float a)
{
    m_color.set( r, g, b, a );
}

static int hexval(char c)
{
    if (c>='0' && c<='9') return c-'0';
    else if (c>='a' && c<='f') return (c-'a')+10;
    else if (c>='A' && c<='F') return (c-'A')+10;
    else return 0;
}


// TODO this could be moved as a shared utility
void Visual3DText::setColor(std::string color)
{
    if (color.empty()) return;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
    if (color[0]>='0' && color[0]<='9')
    {
        sscanf(color.c_str(),"%f %f %f %f", &r, &g, &b, &a);
    }
    else if (color[0]=='#' && color.length()>=7)
    {
        r = (hexval(color[1])*16+hexval(color[2]))/255.0f;
        g = (hexval(color[3])*16+hexval(color[4]))/255.0f;
        b = (hexval(color[5])*16+hexval(color[6]))/255.0f;
        if (color.length()>=9)
            a = (hexval(color[7])*16+hexval(color[8]))/255.0f;
    }
    else if (color[0]=='#' && color.length()>=4)
    {
        r = (hexval(color[1])*17)/255.0f;
        g = (hexval(color[2])*17)/255.0f;
        b = (hexval(color[3])*17)/255.0f;
        if (color.length()>=5)
            a = (hexval(color[4])*17)/255.0f;
    }
    else if (color == "white")    { r = 1.0f; g = 1.0f; b = 1.0f; }
    else if (color == "black")    { r = 0.0f; g = 0.0f; b = 0.0f; }
    else if (color == "red")      { r = 1.0f; g = 0.0f; b = 0.0f; }
    else if (color == "green")    { r = 0.0f; g = 1.0f; b = 0.0f; }
    else if (color == "blue")     { r = 0.0f; g = 0.0f; b = 1.0f; }
    else if (color == "cyan")     { r = 0.0f; g = 1.0f; b = 1.0f; }
    else if (color == "magenta")  { r = 1.0f; g = 0.0f; b = 1.0f; }
    else if (color == "yellow")   { r = 1.0f; g = 1.0f; b = 0.0f; }
    else if (color == "gray")     { r = 0.5f; g = 0.5f; b = 0.5f; }
    else
    {
        serr << "Unknown color "<<color<<sendl;
        return;
    }
    setColor(r,g,b,a);
}


} // namespace visualmodel

} // namespace component

} // namespace sofa

