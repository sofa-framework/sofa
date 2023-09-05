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
#include <sofa/component/visual/TrailRenderer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/AnimateEndEvent.h>

namespace sofa::component::visual
{

template <class DataTypes>
TrailRenderer<DataTypes>::TrailRenderer()
    : core::visual::VisualModel()
    , d_position(initData(&d_position, "position", "Position of the particles behind which a trail is rendered"))
    , d_nbSteps(initData(&d_nbSteps, 100u, "nbSteps", "Number of time steps to use to render the trail"))
    , d_color(initData(&d_color, type::RGBAColor::green(), "color", "Color of the trail"))
    , d_thickness(initData(&d_thickness, 1.f, "thickness", "Thickness of the trail"))
{
    f_listening.setValue(true);
}

template <class DataTypes>
void TrailRenderer<DataTypes>::handleEvent(core::objectmodel::Event* event)
{
    if (sofa::simulation::AnimateEndEvent::checkEventType(event))
    {
        storeParticlePositions();
    }
}

template <class DataTypes>
void TrailRenderer<DataTypes>::removeFirstElements()
{
    const auto nbSteps = d_nbSteps.getValue();

    for (auto& queue : m_trail)
    {
        while (queue.size() > nbSteps)
        {
            queue.erase(queue.begin());
        }
    }
}

template <class DataTypes>
void TrailRenderer<DataTypes>::storeParticlePositions()
{
    const auto& position = d_position.getValue();

    if (position.size() > m_trail.size())
        m_trail.resize(position.size());

    sofa::Size i {};
    for (const auto& pos : position)
    {
        m_trail[i++].push_back(DataTypes::getCPos(pos));
    }

    removeFirstElements();
}

template <class DataTypes>
void TrailRenderer<DataTypes>::doDrawVisual(const core::visual::VisualParams* vparams)
{
    helper::visual::DrawTool*& drawTool = vparams->drawTool();

    const sofa::type::RGBAColor& color = d_color.getValue();
    const float thickness = d_thickness.getValue();

    for (auto& queue : m_trail)
    {
        drawTool->drawLineStrip(queue, thickness, color);
    }
}

template <class DataTypes>
void TrailRenderer<DataTypes>::reset()
{
    m_trail.clear();
}

}
