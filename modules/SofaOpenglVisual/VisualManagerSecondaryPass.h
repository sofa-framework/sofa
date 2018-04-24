/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/*
 * VisualManagerSecondaryPass.h
 *
 *  Created on: 18 fev. 2012
 *      Author: Jeremy Ringard
 */
#ifndef SOFA_COMPONENT_VISUALMANAGER_SECONDARY_PASS_H
#define SOFA_COMPONENT_VISUALMANAGER_SECONDARY_PASS_H
#include "config.h"

#include <SofaOpenglVisual/VisualManagerPass.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <SofaOpenglVisual/OglShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief Render pass element: render the relevant tagged objects in a FBO
 */

class SOFA_OPENGL_VISUAL_API VisualManagerSecondaryPass : public component::visualmodel::VisualManagerPass
{
public:
    SOFA_CLASS(VisualManagerSecondaryPass, component::visualmodel::VisualManagerPass);

    Data< sofa::core::objectmodel::TagSet > input_tags; ///< list of input passes used as source textures
    Data< sofa::core::objectmodel::TagSet > output_tags; ///< output reference tag (use it if the resulting fbo is used as a source for another secondary pass)
    sofa::core::objectmodel::DataFileName fragFilename;

protected:
    OglShader::SPtr m_shaderPostproc;
    SingleLink<VisualManagerSecondaryPass, OglShader, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_shader;

    VisualManagerSecondaryPass();
    virtual ~VisualManagerSecondaryPass();

    virtual void traceFullScreenQuad();

public:
    void init() override;
    void initVisual() override;

    void preDrawScene(core::visual::VisualParams* vp) override;
    bool drawScene(core::visual::VisualParams* vp) override;

    void bindInput(core::visual::VisualParams* /*vp*/);
    void unbindInput();

    helper::gl::FrameBufferObject& getFBO() override {return *fbo;}

    const sofa::core::objectmodel::TagSet& getOutputTags() {return output_tags.getValue();}

private:

    void initShaderInputTexId();
    int nbFbo;
};

}//namespace visualmodel

}//namespace component

}//namespace sofa


#endif // SOFA_CORE_VISUAL_VISUALMANAGER_H
