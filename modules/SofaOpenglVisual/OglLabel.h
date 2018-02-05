/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <string>

#include "config.h"

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/Vec.h>
#include <SofaGraphComponent/BackgroundSetting.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API OglLabel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglLabel, core::visual::VisualModel);

public:
    Data<std::string>            d_prefix;
    Data<std::string>            d_label;
    Data<std::string>            d_suffix;
    Data<unsigned int>           d_x;
    Data<unsigned int>           d_y;
    Data<unsigned int>           d_fontsize;
    Data<defaulttype::RGBAColor> d_color;
    Data<bool>                   d_selectContrastingColor ;
    Data<unsigned int>           d_updateLabelEveryNbSteps;
    Data<bool>                   d_visible;

    void init() override;
    void reinit() override;
    void updateVisual() override;
    void drawVisual(const core::visual::VisualParams* vparams) override;

    void handleEvent(core::objectmodel::Event *) override;

    void parse(core::objectmodel::BaseObjectDescription *arg) override;
    void setColor(float r, float g, float b, float a) ;


protected:
    OglLabel();
    virtual ~OglLabel() {}

    unsigned int                 m_stepCounter;

private:
    std::string                  m_internalLabel;
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
