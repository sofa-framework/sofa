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

#include <sofa/gl/component/shader/config.h>

#include <sofa/gl/component/shader/CompositingVisualLoop.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/gl/FrameBufferObject.h>
#include <sofa/gl/component/shader/OglShader.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::gl::component::shader
{

/**
 *  \brief Render pass element: render the relevant tagged objects in a FBO
 */

class SOFA_GL_COMPONENT_SHADER_API VisualManagerPass : public core::visual::VisualManager
{
public:
    SOFA_CLASS(VisualManagerPass, core::visual::VisualManager);

    Data<float> factor; ///< set the resolution factor for the output pass. default value:1.0
    Data<bool> renderToScreen; ///< if true, this pass will be displayed on screen (only one renderPass in the scene must be defined as renderToScreen)
    Data<std::string> outputName; ///< name the output texture
protected:
    bool checkMultipass(sofa::core::objectmodel::BaseContext* con);
    bool multiPassEnabled;

    std::unique_ptr<sofa::gl::FrameBufferObject> fbo;
    bool prerendered;

    GLint passWidth;
    GLint passHeight;
public:
    VisualManagerPass();
    ~VisualManagerPass() override;


    void init() override;
    void initVisual() override;

    void preDrawScene(core::visual::VisualParams* vp) override;
    bool drawScene(core::visual::VisualParams* vp) override;
    void postDrawScene(core::visual::VisualParams* vp) override;


    void draw(const core::visual::VisualParams* vparams) override;
    void fwdDraw(core::visual::VisualParams*) override;
    void bwdDraw(core::visual::VisualParams*) override;

    void handleEvent(sofa::core::objectmodel::Event* /*event*/) override;

    virtual bool isPrerendered() {return prerendered;}

    virtual sofa::gl::FrameBufferObject& getFBO() {return *fbo;}
    bool hasFilledFbo();
    std::string getOutputName();
};

} // namespace sofa::gl::component::shader
