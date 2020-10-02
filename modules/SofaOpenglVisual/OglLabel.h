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
#ifndef SOFA_COMPONENT_VISUALMODEL_LABEL_H
#define SOFA_COMPONENT_VISUALMODEL_LABEL_H
#include "config.h"

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/Vec.h>
#include <string>
#include <SofaGraphComponent/BackgroundSetting.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API OglLabel : public sofa::core::visual::VisualModel
{
public:
    SOFA_CLASS(OglLabel, sofa::core::visual::VisualModel);

protected:
    OglLabel();
    virtual ~OglLabel() {
    }

    unsigned int stepCounter;

public:

    Data<std::string> prefix;
    Data<std::string> label;
    Data<std::string> suffix;
    Data<unsigned int> x;
    Data<unsigned int> y;
    Data<unsigned int> fontsize;
    Data<std::string> color;
    Data<unsigned int> updateLabelEveryNbSteps;
	Data<bool> f_visible;

    void init();
    void reinit();
    void updateVisual();
    void drawVisual(const core::visual::VisualParams* vparams);

    void handleEvent(sofa::core::objectmodel::Event *);

private:
    void setColor(float r, float g, float b, float a);
    void setColor(std::string color);


    float r,g,b,a;
    typedef defaulttype::Vec4f Color;   // Color with alpha value

    std::string internalLabel;

    sofa::component::configurationsetting::BackgroundSetting* backgroundSetting;


};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
