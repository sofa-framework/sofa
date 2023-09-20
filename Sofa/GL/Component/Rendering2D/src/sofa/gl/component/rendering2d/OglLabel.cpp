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

#include <sofa/gl/component/rendering2d/OglLabel.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/AnimateBeginEvent.h>

#include <string>
#include <iostream>

namespace sofa::gl::component::rendering2d
{

using sofa::component::setting::BackgroundSetting ;
using sofa::core::objectmodel::BaseObjectDescription ;
using sofa::type::RGBAColor ;

OglLabel::OglLabel():
   d_prefix(initData(&d_prefix, std::string(""), "prefix", "The prefix of the text to display"))
  ,d_label(initData(&d_label, std::string(""), "label", "The text to display"))
  ,d_suffix(initData(&d_suffix, std::string(""), "suffix", "The suffix of the text to display"))
  ,d_x(initData(&d_x, (unsigned int)10, "x", "The x position of the text on the screen"))
  ,d_y(initData(&d_y, (unsigned int)10, "y", "The y position of the text on the screen"))
  ,d_fontsize(initData(&d_fontsize, (unsigned int)14, "fontsize", "The size of the font used to display the text on the screen"))
  ,d_color(initData(&d_color, sofa::type::RGBAColor::gray(), "color", "The color of the text to display. (default='gray')"))
  ,d_selectContrastingColor(initData(&d_selectContrastingColor, false, "selectContrastingColor", "Overide the color value but one that contrast with the background color"))
  ,d_updateLabelEveryNbSteps(initData(&d_updateLabelEveryNbSteps, (unsigned int)0, "updateLabelEveryNbSteps", "Update the display of the label every nb of time steps"))
  ,m_stepCounter(0)
{
    f_listening.setValue(true);
}

void OglLabel::parse(BaseObjectDescription *arg)
{
    // BACKWARD COMPATIBILITY April 2017
    const char* value = arg->getAttribute("color") ;
    if(value==nullptr || strcmp(value, "contrast")){
        VisualModel::parse(arg);
        return ;
    }

    arg->setAttribute("selectContrastingColor", std::string("true"));
    arg->removeAttribute("color") ;

    VisualModel::parse(arg);

    /// A send the message after the parsing of the base class so that the "name" of the component
    /// is correctly reported in the message.
    msg_deprecated() << "Attribute color='contrast' is deprecated since SOFA v17.06" << msgendl
                     << "Using deprecated attributes may result in lower performance or un-expected behaviors" << msgendl
                     << "To remove this message you need to update your scene by replacing color='contrast' with "
                        " selectContrastingColor='true'" ;

}

void OglLabel::init()
{
    reinit();
}

void OglLabel::reinit()
{
    if( d_selectContrastingColor.isSet() && d_color.isSet() ){
        msg_warning() << "The selectContrastingColor and color attributes are both set. " << msgendl
                      << "The color attribute will be overriden by the contrasting color. ";
    }

    m_internalLabel = d_label.getValue();

    if( d_selectContrastingColor.getValue() ){
        msg_info() << "Automatically select a color to contrast against the background." ;
        BackgroundSetting* backgroundSetting ;
        this->getContext()->getRootContext()->get(backgroundSetting, sofa::core::objectmodel::BaseContext::SearchRoot);
        if (backgroundSetting)
        {
            //in contrast mode, the text color is selected between black or white depending on the background color
            const RGBAColor& backgroundColor = backgroundSetting->color.getValue();
            float yiq = (float)(backgroundColor[0]*255*299 + backgroundColor[1]*255*587 + backgroundColor[2]*255*114);
            yiq /= 1000;
            if (yiq >= 128)
            {
                msg_info() << "Black is selected to display text on this background." ;
                setColor(0,0,0,1);
            }
            else
            {
                msg_info() << "White is selected to display text on this background." ;
                setColor(1,1,1,1);
            }
        }
        else
        {
            msg_info() << "Background setting not found, cannot use contrast on color data (set white instead)." ;
            setColor(1,1,1,1);
        }
    }
}

void OglLabel::updateVisual()
{
    if (!d_updateLabelEveryNbSteps.getValue()) m_internalLabel = d_label.getValue();
}

void OglLabel::handleEvent(sofa::core::objectmodel::Event *event)
{
    if ( dynamic_cast<simulation::AnimateBeginEvent*>(event) )
    {
        if (d_updateLabelEveryNbSteps.getValue())
        {
            m_stepCounter++;
            if(m_stepCounter > d_updateLabelEveryNbSteps.getValue())
            {
                m_stepCounter = 0;
                m_internalLabel = d_label.getValue();
            }
        }
    }
}

void OglLabel::doDrawVisual(const core::visual::VisualParams* vparams)
{
    // Workaround: Disable showWireframe (polygon mode set to true forcefully in VisualModel drawVisual())
    if (vparams->displayFlags().getShowWireFrame())
    {
        vparams->drawTool()->setPolygonMode(0, false);
    }

    vparams->drawTool()->setLightingEnabled(false);

    const std::string text = d_prefix.getValue() + m_internalLabel.c_str() + d_suffix.getValue();

    vparams->drawTool()->writeOverlayText(
        d_x.getValue(), d_y.getValue(), d_fontsize.getValue(),  // x, y, size
        d_color.getValue(),
        text.c_str());

    // restore polygon mode if needed
    if (vparams->displayFlags().getShowWireFrame())
    {
        vparams->drawTool()->setPolygonMode(0, true);
    }
}

void OglLabel::setColor(float r, float g, float b, float a)
{
    d_color.beginEdit()->set(r,g,b,a);
    d_color.endEdit();
}


int OglLabelClass = core::RegisterObject("Display 2D text in the viewport.")
        .add< OglLabel >()
        ;

} // namespace sofa::gl::component::rendering2d
