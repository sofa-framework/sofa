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
#include <sofa/type/RGBAColor.h>

namespace sofa::core::topology
{
    class BaseMeshTopology;
} // namespace sofa::core::topology

namespace sofa::core::behavior
{
    class BaseMechanicalState;
} // namespace sofa::core::behavior

namespace sofa::component::visual
{

/// Draw camera-oriented (billboard) 3D text
class SOFA_COMPONENT_VISUAL_API Visual3DText : public core::visual::VisualModel
{

public:
    SOFA_CLASS(Visual3DText,core::visual::VisualModel);

protected:
    Visual3DText();

public:
    void init() override;

    void reinit() override;

    void doDrawVisual(const core::visual::VisualParams* vparams) override;
    void drawTransparent(const core::visual::VisualParams* vparams) override;

public:
    Data<std::string> d_text; ///< Test to display
    Data<type::Vec3f> d_position; ///< 3d position
    Data<float> d_scale; ///< text scale
    Data<sofa::type::RGBAColor> d_color; ///< text color. (default=[1.0,1.0,1.0,1.0])
    Data<bool> d_depthTest; ///< perform depth test


};

} // namespace sofa::component::visual
