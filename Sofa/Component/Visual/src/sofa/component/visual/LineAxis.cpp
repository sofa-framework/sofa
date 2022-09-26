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

#include <sofa/component/visual/LineAxis.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>


namespace sofa::component::visual
{

constexpr const char * lineAxisDeprecatedName = "OglLineAxis";

int LineAxisClass = core::RegisterObject("Display scene axis")
        .add< LineAxis >()
        .addAlias(lineAxisDeprecatedName)
        ;

using namespace sofa::defaulttype;

void LineAxis::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    if (std::string(arg->getAttribute("type")) == lineAxisDeprecatedName)
    {
        msg_warning(lineAxisDeprecatedName) << lineAxisDeprecatedName << " is deprecated since SOFA v22.12, and has"
            " been replaced by " << this->getClassName() << ". Please modify your scene. Note that the new component "
            << this->getClassName() << " is located in another module (Sofa.Component.Visual). It means that you "
            "probably need to update also the appropriate RequiredPlugin in your scene. For example, in XML, "
            "consider removing the line <RequiredPlugin name=\"Sofa.GL.Component.Rendering3D\"> to replace it by "
            "<RequiredPlugin name=\"Sofa.Component.Visual\">.";
    }
    Inherit1::parse(arg);
}

LineAxis::LineAxis()
    : d_axis(initData(&d_axis, std::string("xyz"),  "axis", "Axis to draw"))
    , d_size(initData(&d_size, 10.f,  "size", "Size of the squared grid"))
    , d_thickness(initData(&d_thickness, 1.f,  "thickness", "Thickness of the lines in the grid"))
    , d_draw(initData(&d_draw, true,  "draw", "Display the grid or not"))
    , m_drawX(true), m_drawY(true), m_drawZ(true)
{}

void LineAxis::init()
{
    Inherit1::init();
    updateVisual();
}

void LineAxis::reinit()
{
    updateVisual();
}

void LineAxis::updateVisual()
{
    const std::string a = d_axis.getValue();

    m_drawX = a.find_first_of("xX")!=std::string::npos;
    m_drawY = a.find_first_of("yY")!=std::string::npos;
    m_drawZ = a.find_first_of("zZ")!=std::string::npos;
}

void LineAxis::drawVisual(const core::visual::VisualParams* vparams)
{
    if (!d_draw.getValue()) return;

    const float s = d_size.getValue();

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    if(m_drawX)
    {
        vparams->drawTool()->drawLine(
            helper::visual::DrawTool::Vector3(-s*0.5f, 0.0f, 0.0f),
            helper::visual::DrawTool::Vector3(s*0.5f, 0.0f, 0.0f),
            helper::visual::DrawTool::RGBAColor(1.0f, 0.0f, 0.0f, 1.0f));
    }

    if(m_drawY)
    {
        vparams->drawTool()->drawLine(
            helper::visual::DrawTool::Vector3(0.0f, -s*0.5f, 0.0f),
            helper::visual::DrawTool::Vector3(0.0f,  s*0.5f, 0.0f),
            helper::visual::DrawTool::RGBAColor(0.0f, 1.0f, 0.0f, 1.0f));
    }

    if(m_drawZ)
    {
        vparams->drawTool()->drawLine(
            helper::visual::DrawTool::Vector3(0.0f, 0.0f, -s*0.5f),
            helper::visual::DrawTool::Vector3(0.0f, 0.0f, s*0.5f),
            helper::visual::DrawTool::RGBAColor(0.0f, 0.0f, 1.0f, 1.0f));
    }



}


} // namespace sofa::component::visual
