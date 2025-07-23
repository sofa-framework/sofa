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
#include <sofa/component/visual/VisualVectorField.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::visual
{

template <class DataTypes>
VisualVectorField<DataTypes>::VisualVectorField()
    : d_position(initData(&d_position, "position", "Starting position of the rendered vectors"))
    , d_vector(initData(&d_vector, "vector", "List of vectors to render"))
    , d_vectorScale(initData(&d_vectorScale, 1.0_sreal, "vectorScale", "Scaling factor applied on vectors for rendering"))
    , d_drawMode(initData(&d_drawMode, VectorFieldDrawMode("Line"), "drawMode", ("Draw mode for the vectors" + VectorFieldDrawMode::dataDescription()).c_str()))
    , d_color(initData(&d_color, sofa::type::RGBAColor::white(), "color", "Color of the vectors"))
{
}

template <class DataTypes>
void VisualVectorField<DataTypes>::doDrawVisual(const core::visual::VisualParams* vparams)
{
    const auto position = sofa::helper::getReadAccessor(d_position);
    const auto vector = sofa::helper::getReadAccessor(d_vector);

    const auto minSize = std::min(position.size(), vector.size());
    const auto vectorScale = d_vectorScale.getValue();

    auto* drawTool = vparams->drawTool();
    const auto drawMode = d_drawMode.getValue();
    const auto color = d_color.getValue();

    for (std::size_t i = 0; i < minSize; ++i)
    {
        const auto& start = position[i];
        const auto end = start + vectorScale * vector[i];

        if (drawMode == VectorFieldDrawMode("Line"))
        {
            drawTool->drawLines(std::vector<type::Vec3>{{start, end}}, 1, color);
        }
        else if (drawMode == VectorFieldDrawMode("Cylinder"))
        {
            const float radius = static_cast<float>(vectorScale * vector[i].norm() / 20.f);
            drawTool->drawCylinder(start, end, radius, color);
        }
        else if (drawMode == VectorFieldDrawMode("Arrow"))
        {
            const float radius = static_cast<float>(vectorScale * vector[i].norm() / 20.f);
            drawTool->drawArrow(start, end, radius, color);
        }
    }
}

template <class DataTypes>
void VisualVectorField<DataTypes>::computeBBox(const core::ExecParams* exec_params, bool cond)
{
    const auto position = sofa::helper::getReadAccessor(d_position);
    const auto vector = sofa::helper::getReadAccessor(d_vector);

    const auto minSize = std::min(position.size(), vector.size());
    if (minSize == 0)
    {
        return;
    }

    const auto& vectorScale = d_vectorScale.getValue();

    auto bbox = sofa::helper::getWriteOnlyAccessor(this->f_bbox);
    for (size_t i = 0; i < minSize; i++)
    {
        const auto& start = position[i];
        const auto end = start + vectorScale * vector[i];

       bbox.wref().include(start);
       bbox.wref().include(end);
    }
}

}  // namespace sofa::component::visual
