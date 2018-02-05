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
#ifndef SOFA_OGLSCENEFRAME_H
#define SOFA_OGLSCENEFRAME_H
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class OglSceneFrame : public core::visual::VisualModel
{

public:
    SOFA_CLASS(OglSceneFrame, VisualModel);

    typedef core::visual::VisualParams::Viewport Viewport;

    Data<bool> drawFrame;
    Data<sofa::helper::OptionsGroup> style;
    Data<sofa::helper::OptionsGroup> alignment;

    OglSceneFrame():
        drawFrame(initData(&drawFrame, true,  "draw", "Display the frame or not")),
        style(initData(&style, "style", "Style of the frame")),
        alignment(initData(&alignment, "alignment", "Alignment of the frame in the view"))
    {
        quadratic = NULL;

        sofa::helper::OptionsGroup styleOptions(3,"Arrows", "Cylinders", "CubeCones");
        styleOptions.setSelectedItem(1);
        style.setValue(styleOptions);

        sofa::helper::OptionsGroup alignmentOptions(4,"BottomLeft", "BottomRight", "TopRight", "TopLeft");
        alignmentOptions.setSelectedItem(1);
        alignment.setValue(alignmentOptions);
    }

    virtual void init() override;
    virtual void reinit() override;
    virtual void draw(const core::visual::VisualParams*) override;
    virtual void updateVisual() override;


protected:

    GLUquadricObj *quadratic;

};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif //SOFA_OGLSCENEFRAME_H
