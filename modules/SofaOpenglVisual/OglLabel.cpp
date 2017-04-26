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

#include <SofaOpenglVisual/OglLabel.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>

#include <string>
#include <iostream>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using sofa::defaulttype::RGBAColor ;
using sofa::component::configurationsetting::BackgroundSetting ;
using sofa::core::objectmodel::BaseObjectDescription ;

SOFA_DECL_CLASS(OglLabel)

int OglLabelClass = core::RegisterObject("A simple visualization for 2D text.")
        .add< OglLabel >()
        ;

OglLabel::OglLabel(): stepCounter(0)
  ,prefix(initData(&prefix, std::string(""), "prefix", "The prefix of the text to display"))
  ,label(initData(&label, std::string(""), "label", "The text to display"))
  ,suffix(initData(&suffix, std::string(""), "suffix", "The suffix of the text to display"))
  ,x(initData(&x, (unsigned int)10, "x", "The x position of the text on the screen"))
  ,y(initData(&y, (unsigned int)10, "y", "The y position of the text on the screen"))
  ,fontsize(initData(&fontsize, (unsigned int)14, "fontsize", "The size of the font used to display the text on the screen"))
  ,color(initData(&color, defaulttype::RGBAColor::fromString("gray"), "color", "The color of the text to display. (default='gray')"))
  ,m_selectContrastingColor(initData(&m_selectContrastingColor, false, "selectContrastingColor", "Overide the color value but one that contrast with the background color"))
  ,updateLabelEveryNbSteps(initData(&updateLabelEveryNbSteps, (unsigned int)0, "updateLabelEveryNbSteps", "Update the display of the label every nb of time steps"))
  ,f_visible(initData(&f_visible,true,"visible","Is label displayed"))
{
    f_listening.setValue(true);
}

void OglLabel::parse(BaseObjectDescription *arg)
{
    // BACKWARD COMPATIBILITY April 2017
    const char* value = arg->getAttribute("color") ;
    if(strcmp(value, "contrast"))
        return ;

    msg_deprecated() << "Attribute color='contrast' is deprecated since Sofa 17.06.  " << msgendl
                     << "Using deprecated attribute may result in lower performance and un-expected behavior" << msgendl
                     << "To remove this message you need to update your scene by replacing color='contrast' with "
                        " selectConstrastingColor='true'" ;

    arg->setAttribute("selectContrastingColor", "true");

    Base::parse(arg);
}

void OglLabel::init()
{
    reinit();
}

void OglLabel::reinit()
{
    internalLabel = label.getValue();

    if( m_selectContrastingColor.getValue() ){
        msg_info() << "Select the color from background." ;
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
                msg_info() << "Black is selected to display text on this background" ;
                setColor(0,0,0,1);
            }
            else
            {
                msg_info() << "White is selected to display text on this background" ;
                setColor(1,1,1,1);
            }
        }
        else
        {
            msg_info() << "Background setting not found, cannot use contrast on color data (set white instead)" ;
            setColor(1,1,1,1);
        }
    }
}

void OglLabel::updateVisual()
{
    if (!updateLabelEveryNbSteps.getValue()) internalLabel = label.getValue();
}

void OglLabel::handleEvent(sofa::core::objectmodel::Event *event)
{
    if ( /*simulation::AnimateEndEvent* ev =*/  dynamic_cast<sofa::simulation::AnimateBeginEvent*>(event))
    {
        if (updateLabelEveryNbSteps.getValue())
        {
            stepCounter++;
            if(stepCounter > updateLabelEveryNbSteps.getValue())
            {
                stepCounter = 0;
                internalLabel = label.getValue();
            }
        }
    }
}

void OglLabel::drawVisual(const core::visual::VisualParams* vparams)
{

    if (!f_visible.getValue() ) return;

    // Save state and disable clipping plane
    glPushAttrib(GL_ENABLE_BIT);
    for(int i = 0; i < GL_MAX_CLIP_PLANES; ++i)
        glDisable(GL_CLIP_PLANE0+i);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_1D);
    glDisable(GL_BLEND);
    glDepthMask(1);

    vparams->drawTool()->setLightingEnabled(false);
    // vparams->drawTool()->setPolygonMode(1,true);

    // color of the text
    glColor4fv( color.getValue().data() );

    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color.getValue().data() );
    static const float emissive[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
    static const float specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
    glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive);
    glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular);
    glMaterialf  (GL_FRONT_AND_BACK, GL_SHININESS, 20);

    std::string text = prefix.getValue() + internalLabel.c_str() + suffix.getValue();

    vparams->drawTool()->writeOverlayText(
        x.getValue(), y.getValue(), fontsize.getValue(),  // x, y, size
        color.getValue(),
        text.c_str());


    // Restore state
    glPopAttrib();
}

void OglLabel::setColor(float r, float g, float b, float a)
{
    color.beginEdit()->set(r,g,b,a);
    color.endEdit();
}

/*
void OglLabel::setColor(std::string scolor)
{
    if (scolor.empty())
        return;

    if(!color.read(scolor)){
        msg_warning() << " '"<< scolor<< "' is not a valid color." ;
    }

    setColor(r,g,b,a);
}*/

} // namespace visualmodel

} // namespace component

} // namespace sofa
