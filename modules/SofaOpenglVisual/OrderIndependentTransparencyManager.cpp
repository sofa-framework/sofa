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
//
// C++ Implementation: OrderIndependentTransparencyManager
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <SofaOpenglVisual/OglModel.h>
#include <SofaOpenglVisual/OrderIndependentTransparencyManager.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>


namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace helper::gl;
using namespace simulation;
using namespace core::visual;

SOFA_DECL_CLASS(OrderIndependentTransparencyManager)
//Register OrderIndependentTransparencyManager in the Object Factory
int OrderIndependentTransparencyManagerClass = core::RegisterObject("OrderIndependentTransparencyManager")
        .add< OrderIndependentTransparencyManager >()
        ;

class VisualOITDrawVisitor : public VisualDrawVisitor
{
public:
    VisualOITDrawVisitor(core::visual::VisualParams* params, GLSLShader* oitShader)
        : VisualDrawVisitor(params)
        , shader(oitShader)
    {
    }

    void processVisualModel(simulation::Node* node, core::visual::VisualModel* vm);

public:
    GLSLShader* shader;

};

OrderIndependentTransparencyManager::OrderIndependentTransparencyManager()
    : depthScale(initData(&depthScale, 0.01f, "depthScale", "Depth scale"))
    , fbo()
    , accumulationShader()
    , compositionShader()
{

}

OrderIndependentTransparencyManager::~OrderIndependentTransparencyManager()
{

}

void OrderIndependentTransparencyManager::init()
{

}

void OrderIndependentTransparencyManager::bwdInit()
{

}

void OrderIndependentTransparencyManager::initVisual()
{
    GLSLShader::InitGLSL();

// accumulation shader

    std::string accumulationVertexShaderFilename = "shaders/orderIndependentTransparency/accumulation.vert";
    helper::system::DataRepository.findFile(accumulationVertexShaderFilename);

    std::string accumulationFragmentShaderFilename = "shaders/orderIndependentTransparency/accumulation.frag";
    helper::system::DataRepository.findFile(accumulationFragmentShaderFilename);

    accumulationShader.InitShaders(accumulationVertexShaderFilename, accumulationFragmentShaderFilename);

// composition shader

    std::string compositionVertexShaderFilename = "shaders/orderIndependentTransparency/composition.vert";
    helper::system::DataRepository.findFile(compositionVertexShaderFilename);

    std::string compositionFragmentShaderFilename = "shaders/orderIndependentTransparency/composition.frag";
    helper::system::DataRepository.findFile(compositionFragmentShaderFilename);

    compositionShader.InitShaders(compositionVertexShaderFilename, compositionFragmentShaderFilename);
    compositionShader.TurnOn();
    compositionShader.SetInt(compositionShader.GetVariable("AccumulationSampler"), 0);
    compositionShader.SetInt(compositionShader.GetVariable("RevealageSampler"), 1);
    compositionShader.TurnOff();
}

void OrderIndependentTransparencyManager::fwdDraw(core::visual::VisualParams* /*vp*/)
{

}

void OrderIndependentTransparencyManager::bwdDraw(core::visual::VisualParams* )
{

}

void OrderIndependentTransparencyManager::draw(const core::visual::VisualParams* )
{
    // debug draw
}

void OrderIndependentTransparencyManager::reinit()
{

}

void OrderIndependentTransparencyManager::preDrawScene(VisualParams* /*vp*/)
{

}

//static void DrawQuad(float offset, float scale = 1.0f)
//{
//    glBegin(GL_QUADS);
//    {
//        glVertex3f(-scale, -scale, offset);
//        glVertex3f( scale, -scale, offset);
//        glVertex3f( scale,  scale, offset);
//        glVertex3f(-scale,  scale, offset);
//    }
//    glEnd();
//}

static void DrawFullScreenQuad()
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(-1.0f, -1.0f);

        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(1.0f, -1.0f);

        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(1.0f, 1.0f);

        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(-1.0f, 1.0f);
    }
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

bool OrderIndependentTransparencyManager::drawScene(VisualParams* vp)
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    if(0 == viewport[2] || 0 == viewport[3])
        return false;

// draw opaques normally

    glClear(GL_DEPTH_BUFFER_BIT);
    glDepthMask(GL_TRUE);

    drawOpaques(vp);

// draw transparents in a fbo

    GLint previousDrawFBO = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFBO);

    // init fbo

    fbo.init(viewport[2], viewport[3]);

    // copy current depth buffer to fbo render buffer

    fbo.copyDepth(previousDrawFBO);

    // bind fbo

    fbo.bind();

    // accumulation

    GLenum buffers[] = {GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT};
    glDrawBuffers(2, buffers);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glDepthMask(GL_FALSE);
    glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);

    accumulationShader.TurnOn();
    accumulationShader.SetFloat(accumulationShader.GetVariable("DepthScale"), depthScale.getValue());
    drawTransparents(vp, &accumulationShader);
    accumulationShader.TurnOff();

// compose

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, previousDrawFBO);
    // TODO: set the previously bound drawBuffers

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    compositionShader.TurnOn();

    fbo.bindTextures();
    DrawFullScreenQuad();
    fbo.releaseTextures();

    compositionShader.TurnOff();

    glDepthMask(GL_TRUE);

    return true;
}

void OrderIndependentTransparencyManager::drawOpaques(VisualParams* vp)
{
    Node* node = dynamic_cast<Node*>(getContext());
    if(!node)
        return;

    vp->pass() = sofa::core::visual::VisualParams::Std;
    VisualDrawVisitor drawStandardVisitor(vp);
    drawStandardVisitor.setTags(this->getTags());
    node->execute(&drawStandardVisitor);

//    glColor4f(0.75f, 0.75f, 0.75f, 1.0f);
//    DrawQuad(-2.0f, 10.0f);
}

void OrderIndependentTransparencyManager::drawTransparents(VisualParams* vp, GLSLShader* oitShader)
{
    Node* node = dynamic_cast<Node*>(getContext());
    if(!node)
        return;

    vp->pass() = sofa::core::visual::VisualParams::Transparent;
    VisualOITDrawVisitor drawTransparentVisitor(vp, oitShader);
    drawTransparentVisitor.setTags(this->getTags());
    node->execute(&drawTransparentVisitor);

//    glColor4f(0.0f, 0.0f, 1.0f, 0.6f);
//    DrawQuad(1.0f);

//    glColor4f(1.0f, 1.0f, 0.0f, 0.6f);
//    DrawQuad(0.0f);

//    glColor4f(1.0f, 0.0f, 0.0f, 0.6f);
//    DrawQuad(-1.0f);
}

void OrderIndependentTransparencyManager::postDrawScene(VisualParams* /*vp*/)
{
    // TODO: restore default parameter if any
}

OrderIndependentTransparencyManager::FrameBufferObject::FrameBufferObject()
    : id(0)
    , width(0)
    , height(0)
    , depthRenderbuffer(0)
    , accumulationTexture(0)
    , revealageTexture(0)
{

}

void OrderIndependentTransparencyManager::FrameBufferObject::init(int w, int h)
{
    if(0 == w || 0 == h)
        return;

    if(w == width && h == height)
        return;

    destroy();

    width = w;
    height = h;

// fbo

    glGenFramebuffersEXT(1, &id);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, id);

// depth render buffer

    glGenRenderbuffersEXT(1, &depthRenderbuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthRenderbuffer);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH24_STENCIL8_EXT, width, height);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthRenderbuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

// accumulation texture

    glGenTextures(1, &accumulationTexture);
    glBindTexture(GL_TEXTURE_RECTANGLE, accumulationTexture);
    glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE, accumulationTexture, 0);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

// revealage texture

    glGenTextures(1, &revealageTexture);
    glBindTexture(GL_TEXTURE_RECTANGLE, revealageTexture);
    glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_R16F, width, height, 0, GL_RED, GL_FLOAT, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_RECTANGLE, revealageTexture, 0);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void OrderIndependentTransparencyManager::FrameBufferObject::destroy()
{
    glDeleteTextures(1, &revealageTexture);
    glDeleteTextures(1, &accumulationTexture);
    glDeleteRenderbuffers(1, &depthRenderbuffer);

    glDeleteFramebuffersEXT(1, &id);

    id = 0;
}

void OrderIndependentTransparencyManager::FrameBufferObject::copyDepth(GLuint fromFBO)
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fromFBO);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, id);
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
}

void OrderIndependentTransparencyManager::FrameBufferObject::bind()
{
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, id);
}

void OrderIndependentTransparencyManager::FrameBufferObject::bindTextures()
{
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE, revealageTexture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE, accumulationTexture);
}

void OrderIndependentTransparencyManager::FrameBufferObject::releaseTextures()
{
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
}

void VisualOITDrawVisitor::processVisualModel(simulation::Node* node, core::visual::VisualModel* vm)
{
    bool hasTexture = false;

    OglModel* oglModel = dynamic_cast<OglModel*>(vm);
    if(oglModel)
    {
        oglModel->blendTransparency.setValue(false);
        hasTexture = oglModel->hasTexture();
    }

    GLSLShader* oitShader = 0;

    sofa::core::visual::Shader* nodeShader = NULL;
    if(hasShader) // has custom oit shader
    {
        nodeShader = node->getShader(subsetsToManage);

        OglOITShader* oglOITShader = dynamic_cast<OglOITShader*>(nodeShader);
        if(oglOITShader)
            oitShader = oglOITShader->accumulationShader();
    }

    if(!oitShader)
        oitShader = shader;

    oitShader->TurnOn();
    oitShader->SetInt(oitShader->GetVariable("HasTexture"), hasTexture ? 1 : 0);
    vm->drawTransparent(vparams);
}

} // namespace visualmodel

} // namespace component

} // namespace sofa
