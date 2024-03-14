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
#include <sofa/component/visual/config.h>

#include <sofa/core/visual/VisualModel.h>

namespace sofa::component::visual
{

class SOFA_COMPONENT_VISUAL_API LineAxis : public core::visual::VisualModel
{
public:
    SOFA_CLASS(LineAxis, VisualModel);

    Data<std::string> d_axis; ///< Axis to draw
    Data<float> d_size; ///< Size of the squared grid
    Data<float> d_thickness; ///< Thickness of the lines in the grid
    core::objectmodel::lifecycle::RemovedData d_draw {this, "v23.06", "23.12", "draw", "Use the 'enable' data field instead of 'draw'"};

    LineAxis();

    void init() override;
    void reinit() override;
    void doDrawVisual(const core::visual::VisualParams*) override;
    void updateVisual() override;

protected:

    bool m_drawX;
    bool m_drawY;
    bool m_drawZ;

};

} // namespace sofa::component::visual
