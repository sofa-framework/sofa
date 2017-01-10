/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
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
  ,updateLabelEveryNbSteps(initData(&updateLabelEveryNbSteps, (unsigned int)0, "updateLabelEveryNbSteps", "Update the display of the label every nb of time steps"))
  ,f_visible(initData(&f_visible,true,"visible","Is label displayed"))
{
    f_listening.setValue(true);
}

void OglLabel::init()
{
    reinit();
}


void OglLabel::reinit()
{
    internalLabel = label.getValue();
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
    glColor4fv( color.getValue().ptr() );

    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color.getValue().ptr() );
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

} // namespace visualmodel

} // namespace component

} // namespace sofa
