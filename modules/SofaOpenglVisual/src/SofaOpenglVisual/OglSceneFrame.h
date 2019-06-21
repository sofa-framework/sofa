/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

    Data<bool> drawFrame; ///< Display the frame or not
    Data<sofa::helper::OptionsGroup> style; ///< Style of the frame
    Data<sofa::helper::OptionsGroup> alignment; ///< Alignment of the frame in the view

    OglSceneFrame();

    void init() override;
    void reinit() override;
    void draw(const core::visual::VisualParams*) override;
    void updateVisual() override;


protected:

    GLUquadricObj *quadratic;

};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif //SOFA_OGLSCENEFRAME_H
