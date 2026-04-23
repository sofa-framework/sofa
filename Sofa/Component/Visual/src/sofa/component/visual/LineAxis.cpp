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
#include <sofa/helper/narrow_cast.h>


namespace sofa::component::visual
{

using helper::visual::DrawTool;

void registerLineAxis(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Display scene axis")
        .add< LineAxis >());
}

using namespace sofa::defaulttype;

LineAxis::LineAxis()
    : d_axis(initData(&d_axis, std::string("xyz"),  "axis", "Axis to draw."))
    , d_size(initData(&d_size, 10.f,  "size", "Size of the lines."))
    , d_infinite(initData(&d_infinite, false,  "infinite", "If true, ignore the 'size' and draw infinite lines."))
    , d_thickness(initData(&d_thickness, 1.f,  "thickness", "Thickness of the lines."))
    , d_vanishing(initData(&d_vanishing, false,  "vanishing", "In case of infinite lines, should the lines gradually vanish."))
    , d_nbLineSegment(initData(&d_nbLineSegment, 1,  "nbLineSegments", "Number of line segments used to draw each axis line. Higher values reduce rendering artifacts, especially when using vertex-based OpenGL effects."))
    , m_drawX(true), m_drawY(true), m_drawZ(true)
{}

void LineAxis::init()
{
    Inherit1::init();
    updateLine();
}

void LineAxis::reinit()
{
    updateLine();
}

void LineAxis::doUpdateVisual(const core::visual::VisualParams*)
{
    updateLine();
}

void LineAxis::updateLine()
{
    const std::string a = d_axis.getValue();

    m_drawX = a.find_first_of("xX")!=std::string::npos;
    m_drawY = a.find_first_of("yY")!=std::string::npos;
    m_drawZ = a.find_first_of("zZ")!=std::string::npos;
}

void LineAxis::doDrawVisual(const core::visual::VisualParams* vparams)
{
    const double s = sofa::helper::narrow_cast<double>(d_size.getValue());
    const double hs = s * 0.5; // half the size

    auto drawtool = vparams->drawTool();
    drawtool->disableLighting();

    const unsigned int nl = d_nbLineSegment.getValue();
    const float ls = s / nl;

    std::vector<type::Vec3> points;
    points.resize(nl+1);
    std::vector<type::Vec2i> indices;
    indices.resize(nl);

    const auto& bbox = helper::getReadAccessor(getContext()->f_bbox);
    auto v = bbox->maxBBox() - bbox->minBBox();
    const auto& thickness = helper::getReadAccessor(d_thickness);
    const auto& vanishing = helper::getReadAccessor(d_vanishing);
    const auto& infinite = helper::getReadAccessor(d_infinite);

    if(m_drawX)
    {
        for (unsigned int j = 0 ; j < nl; ++j)
        {
            points[j] = DrawTool::Vec3(-hs + ls*j, 0.0, 0.0);
            points[j+1] = DrawTool::Vec3(-hs + ls*(j+1), 0.0, 0.0);
            indices[j] = type::Vec2i(j, j+1);
        }

        if (!infinite)
        {
            drawtool->drawLines(points, indices, thickness, DrawTool::RGBAColor::red());
        }
        else // infinite line
        {
            drawtool->drawInfiniteLine(DrawTool::Vec3(0, 0, 0), DrawTool::Vec3(v.x(), 0, 0), thickness,
                                                  DrawTool::RGBAColor::red(), vanishing);
            drawtool->drawInfiniteLine(DrawTool::Vec3(0, 0, 0), DrawTool::Vec3(-v.x(), 0, 0), thickness,
                                                  DrawTool::RGBAColor::red(), vanishing);
        }
    }

    if(m_drawY)
    {
        for (unsigned int j = 0 ; j < nl; ++j)
        {
            points[j] = DrawTool::Vec3(0.0, -hs + ls*j, 0.0);
            points[j+1] = DrawTool::Vec3(0.0, -hs + ls*(j+1), 0.0);
            indices[j] = type::Vec2i(j, j+1);
        }

        if (!infinite)
        {
            drawtool->drawLines(points, indices, thickness, DrawTool::RGBAColor::green());
        }
        else // infinite line
        {
            drawtool->drawInfiniteLine(DrawTool::Vec3(0, 0, 0), DrawTool::Vec3(0, v.y(), 0), thickness,
                                                  DrawTool::RGBAColor::green(), vanishing);
            drawtool->drawInfiniteLine(DrawTool::Vec3(0, 0, 0), DrawTool::Vec3(0, -v.y(), 0), thickness,
                                                  DrawTool::RGBAColor::green(), vanishing);
        }
    }

    if(m_drawZ)
    {
        for (unsigned int j = 0 ; j < nl; ++j)
        {
            points[j] = DrawTool::Vec3(0.0, 0.0, -hs + ls*j);
            points[j+1] = DrawTool::Vec3(0.0, 0.0, -hs + ls*(j+1));
            indices[j] = type::Vec2i(j, j+1);
        }

        if (!infinite)
        {
            drawtool->drawLines(points, indices, thickness, DrawTool::RGBAColor::blue());
        }
        else // infinite line
        {
            drawtool->drawInfiniteLine(DrawTool::Vec3(0, 0, 0), DrawTool::Vec3(0, 0, v.z()), thickness,
                                                  DrawTool::RGBAColor::blue(), vanishing);
            drawtool->drawInfiniteLine(DrawTool::Vec3(0, 0, 0), DrawTool::Vec3(0, 0, -v.z()), thickness,
                                                  DrawTool::RGBAColor::blue(), vanishing);
        }
    }
}


} // namespace sofa::component::visual
