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
#include <sofa/component/visual/VisualPointCloud.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::visual
{

template <class DataTypes>
VisualPointCloud<DataTypes>::VisualPointCloud()
    : d_position(initData(&d_position, "position", "The position of the points to display"))
    , d_drawMode(initData(&d_drawMode, "drawMode", ("The draw mode:\n" + DrawMode::dataDescription()).c_str()))
    , d_pointSize(initData(&d_pointSize, 1.f, "pointSize", "The size of the points and frames"))
    , d_sphereRadius(initData(&d_sphereRadius, "sphereRadius", "The radius list of the spheres"))
    , d_color(initData(&d_color, "color", "The color of the points"))
    , d_showIndices(initData(&d_showIndices, false, "showIndices", "Show the indices of the points"))
    , d_indicesScale(initData(&d_indicesScale, 1.f, "indicesScale", "The scale of the indices"))
    , d_indicesColor(initData(&d_indicesColor, "indicesColor", "The color of the indices"))
{
}

template <class DataTypes>
typename VisualPointCloud<DataTypes>::DrawMode VisualPointCloud<DataTypes>::defaultDrawMode()
{
    if constexpr (hasWriteOpenGlMatrix<DataTypes>)
        return DrawMode("Frame");
    else
        return DrawMode("Point");
}


template <class DataTypes>
void VisualPointCloud<DataTypes>::computeBBox(const core::ExecParams* exec_params, bool onlyVisible)
{
    SOFA_UNUSED(exec_params);
    if(!onlyVisible)
        return;

    const auto position = sofa::helper::getReadAccessor(d_position);
    const size_t positionSize = position.size();

    if (positionSize <= 0) return;

    type::Vec3 pvec3;
    type::BoundingBox bbox;
    for (const auto& p : position)
    {
        DataTypes::get(pvec3[0], pvec3[1], pvec3[2], p);
        bbox.include(pvec3);
    }

    this->f_bbox.setValue(bbox);
}

template <class DataTypes>
type::vector<type::Vec3> VisualPointCloud<DataTypes>::convertCoord() const
{
    const auto position = sofa::helper::getReadAccessor(d_position);

    type::vector<type::Vec3> displayedPoints;
    displayedPoints.reserve(position.size());
    for (const auto& point : position)
    {
        displayedPoints.push_back(DataTypes::getCPos(point));
    }

    return displayedPoints;
}

template <class DataTypes>
void VisualPointCloud<DataTypes>::doDrawVisual(const core::visual::VisualParams* vparams)
{
    const auto drawMode = d_drawMode.getValue();
    auto* drawTool = vparams->drawTool();

    const auto color = d_color.getValue();

    if (drawMode == DrawMode("Point") || drawMode == DrawMode("Sphere"))
    {
        if (drawMode == DrawMode("Point"))
        {
            drawTool->setLightingEnabled(false);
            drawTool->drawPoints(convertCoord(), d_pointSize.getValue(), color);
        }
        else if (drawMode == DrawMode("Sphere"))
        {
            auto radius = sofa::helper::getWriteAccessor(d_sphereRadius);
            const auto displayedPoints = convertCoord();

            float defaultRadius = 1.f;
            if (!radius.empty())
            {
                defaultRadius = radius.back();
            }
            radius.resize(displayedPoints.size(), defaultRadius);

            drawTool->setLightingEnabled(true);
            drawTool->drawFakeSpheres(displayedPoints, radius.ref(), color);
        }
    }
    else
    {
        if constexpr (hasWriteOpenGlMatrix<DataTypes>)
        {
            if (drawMode == DrawMode("Frame"))
            {
                drawFrames(vparams, color);
            }
        }
    }

    if (d_showIndices.getValue())
    {
        drawIndices(vparams);
    }
}

template <class DataTypes>
void VisualPointCloud<DataTypes>::drawIndices(const core::visual::VisualParams* vparams) const
{
    const float scale = static_cast<float>(
        (vparams->sceneBBox().maxBBox() - vparams->sceneBBox().minBBox()).norm() *
        d_indicesScale.getValue());
    vparams->drawTool()->draw3DText_Indices(convertCoord(), scale, d_indicesColor.getValue());
}

template <class DataTypes>
void VisualPointCloud<DataTypes>::drawFrames(const core::visual::VisualParams* vparams,
                                             type::RGBAColor color) requires hasWriteOpenGlMatrix<DataTypes>
{
    if constexpr (hasWriteOpenGlMatrix<DataTypes>)
    {
        auto* drawTool = vparams->drawTool();
        const auto position = sofa::helper::getReadAccessor(d_position);

        const auto pointSize = d_pointSize.getValue();
        const bool isColorSet = d_color.isSet();
        const type::Vec3f sizes(1.0f, 1.0f, 1.0f);

        for (const auto& point : position)
        {
            drawTool->pushMatrix();

            float glTransform[16];
            point.writeOpenGlMatrix(glTransform);

            drawTool->multMatrix(glTransform);
            drawTool->scale(pointSize);

            if (isColorSet)
            {
                drawTool->drawFrame(type::Vec3(), type::Quat<SReal>(), sizes);
            }
            else
            {
                drawTool->drawFrame(type::Vec3(), type::Quat<SReal>(),sizes);
            }
            drawTool->popMatrix();
        }
    }
}

}  // namespace sofa::component::visual
