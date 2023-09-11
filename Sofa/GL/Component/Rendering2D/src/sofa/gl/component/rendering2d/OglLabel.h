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
#include <sofa/gl/component/rendering2d/config.h>

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/type/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/gl/template.h>
#include <sofa/type/Vec.h>
#include <sofa/component/setting/BackgroundSetting.h>

namespace sofa::gl::component::rendering2d
{

class SOFA_GL_COMPONENT_RENDERING2D_API OglLabel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglLabel, core::visual::VisualModel);

public:
    Data<std::string>            d_prefix; ///< The prefix of the text to display
    Data<std::string>            d_label; ///< The text to display
    Data<std::string>            d_suffix; ///< The suffix of the text to display
    Data<unsigned int>           d_x; ///< The x position of the text on the screen
    Data<unsigned int>           d_y; ///< The y position of the text on the screen
    Data<unsigned int>           d_fontsize; ///< The size of the font used to display the text on the screen
    Data<sofa::type::RGBAColor> d_color; ///< The color of the text to display. (default='gray')
    Data<bool>                   d_selectContrastingColor ; ///< Overide the color value but one that contrast with the background color
    Data<unsigned int>           d_updateLabelEveryNbSteps; ///< Update the display of the label every nb of time steps
    core::objectmodel::lifecycle::RemovedData d_visible {this, "v23.06", "23.12", "visible", "Use the 'enable' data field instead of 'visible'"};


    void init() override;
    void reinit() override;
    void updateVisual() override;
    void doDrawVisual(const core::visual::VisualParams* vparams) override;

    void handleEvent(core::objectmodel::Event *) override;

    void parse(core::objectmodel::BaseObjectDescription *arg) override;
    void setColor(float r, float g, float b, float a) ;


protected:
    OglLabel();
    ~OglLabel() override {}

    unsigned int                 m_stepCounter;

private:
    std::string                  m_internalLabel;
};

} // namespace sofa::gl::component::rendering2d
