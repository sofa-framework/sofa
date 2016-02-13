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
/*
 * VisualManagerSecondaryPass.h
 *
 *  Created on: 18 fev. 2012
 *      Author: Jeremy Ringard
 */

#include <SofaOpenglVisual/VisualManagerSecondaryPass.h>
#include <sofa/simulation/common/Node.h>
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

SOFA_DECL_CLASS(VisualManagerSecondaryPass)
//Register LightManager in the Object Factory
int VisualManagerSecondaryPassClass = core::RegisterObject("VisualManagerSecondaryPass")
        .add< VisualManagerSecondaryPass >()
        ;

VisualManagerSecondaryPass::VisualManagerSecondaryPass()
    :input_tags(initData( &input_tags, "input_tags", "list of input passes used as source textures")),
     output_tags(initData( &output_tags, "output_tags", "output reference tag (use it if the resulting fbo is used as a source for another secondary pass)")),
     fragFilename(initData(&fragFilename, "fragFilename", "Set the fragment shader filename to load"))
{
    nbFbo=0;
    prerendered=false;
}

VisualManagerSecondaryPass::~VisualManagerSecondaryPass()
{
}

void VisualManagerSecondaryPass::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    multiPassEnabled=checkMultipass(context);
    fbo = new FrameBufferObject(true, true, true);
}

void VisualManagerSecondaryPass::initVisual()
{
    shader_postproc = sofa::core::objectmodel::New<OglShader>();
    shader_postproc->vertFilename.setValueAsString("shaders/compositing.vert");

    if(fragFilename.getValue().empty())
    {
        std::cerr << "fragFilename attribute shall not be null. Using compositing.frag instead" << std::endl;
        shader_postproc->fragFilename.setValueAsString("shaders/compositing.frag");
    }
    else
        shader_postproc->fragFilename.setValueAsString(fragFilename.getFullPath());

    shader_postproc->init();
    shader_postproc->initVisual();
    initShaderInputTexId();

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    passWidth = (GLint)(viewport[2]*factor.getValue());
    passHeight = (GLint)(viewport[3]*factor.getValue());

    fbo->init(passWidth, passHeight);
}

void VisualManagerSecondaryPass::initShaderInputTexId()
{
    nbFbo=0;

    sofa::simulation::Node* gRoot = dynamic_cast<simulation::Node*>(this->getContext());
    sofa::simulation::Node::Sequence<core::visual::VisualManager>::iterator begin = gRoot->visualManager.begin();
    sofa::simulation::Node::Sequence<core::visual::VisualManager>::iterator end = gRoot->visualManager.end();
    sofa::simulation::Node::Sequence<core::visual::VisualManager>::iterator it;
    for (it = begin; it != end; ++it)
    {
        VisualManagerPass *currentPass=dynamic_cast<VisualManagerPass*>((*it));
        VisualManagerSecondaryPass *currentSecondaryPass=dynamic_cast<VisualManagerSecondaryPass*>((*it));
        if(currentSecondaryPass && (this->getName()!=currentSecondaryPass->getName()))
        {
            if((!currentSecondaryPass->getOutputTags().empty()) && (input_tags.getValue().includes(currentSecondaryPass->getOutputTags())) )
            {
                shader_postproc->setInt(0, (currentSecondaryPass->getOutputName()).c_str(), nbFbo);
                //std::cout << "---"<<this->getName()<<"--- add sampler2D \""<< currentSecondaryPass->getName()<<"\" at id="<<nbFbo<<std::endl;
                nbFbo++;
            }
        }
        else
        {
            if(currentPass && (this->getName()!=currentPass->getName()))
            {
                if(input_tags.getValue().includes(currentPass->getTags()))
                {
                    shader_postproc->setInt(0, (currentPass->getOutputName()).c_str(), nbFbo);
                    //std::cout << "---"<<this->getName()<<"--- add sampler2D \""<< currentPass->getName()<<"\" at id="<<nbFbo<<std::endl;
                    nbFbo++;
                    shader_postproc->setInt(0, (currentPass->getOutputName()+"_Z").c_str(), nbFbo);
                    //std::cout << "---"<<this->getName()<<"--- add sampler2D \""<< currentPass->getName()<<"_Z\" at id="<<nbFbo<<std::endl;
                    nbFbo++;
                }
            }
        }
    }
    nbFbo=0;
}

void VisualManagerSecondaryPass::preDrawScene(core::visual::VisualParams* vp)
{
    if(renderToScreen.getValue() || (!multiPassEnabled))
        return;

    shader_postproc->setFloat(0,"zNear", (float)vp->zNear());
    shader_postproc->setFloat(0,"zFar", (float)vp->zFar());
    shader_postproc->setInt(0,"width", /*passWidth*/vp->viewport()[2]); //not sure of the value I should put here...
    shader_postproc->setInt(0,"height", /*passHeight*/vp->viewport()[3]);


    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    //todo: bind input textures
    bindInput(vp);

    shader_postproc->start();


    fbo->start();

    glViewport(0,0,passWidth,passHeight);
    traceFullScreenQuad();

    fbo->stop();
    shader_postproc->stop();

    //todo: unbind input textures
    unbindInput();

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    prerendered=true;
}

void VisualManagerSecondaryPass::traceFullScreenQuad()
{
    float vxmax, vymax ;
    float vxmin, vymin ;
    float txmax,tymax;
    float txmin,tymin;

    txmin = tymin = 0.0;
    vxmin = vymin = -1.0;
    vxmax = vymax = txmax = tymax = 1.0;

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin,tymax,0.0); glVertex3f(vxmin,vymax,0.0);
        glTexCoord3f(txmax,tymax,0.0); glVertex3f(vxmax,vymax,0.0);
        glTexCoord3f(txmax,tymin,0.0); glVertex3f(vxmax,vymin,0.0);
        glTexCoord3f(txmin,tymin,0.0); glVertex3f(vxmin,vymin,0.0);
    }
    glEnd();
}

void VisualManagerSecondaryPass::bindInput(core::visual::VisualParams* /*vp*/)
{
    nbFbo=0;

    sofa::simulation::Node* gRoot = dynamic_cast<simulation::Node*>(this->getContext());
    sofa::simulation::Node::Sequence<core::visual::VisualManager>::iterator begin = gRoot->visualManager.begin();
    sofa::simulation::Node::Sequence<core::visual::VisualManager>::iterator end = gRoot->visualManager.end();
    sofa::simulation::Node::Sequence<core::visual::VisualManager>::iterator it;
    for (it = begin; it != end; ++it)
    {
        VisualManagerPass *currentPass=dynamic_cast<VisualManagerPass*>((*it));
        VisualManagerSecondaryPass *currentSecondaryPass=dynamic_cast<VisualManagerSecondaryPass*>((*it));
        if(currentSecondaryPass && (this->getName()!=currentSecondaryPass->getName()))
        {
            if((!currentSecondaryPass->getOutputTags().empty()) && (input_tags.getValue().includes(currentSecondaryPass->getOutputTags())) )
            {
                if (!currentSecondaryPass->hasFilledFbo())
                {
                    std::cerr << "Error: SecondaryPass \"" << this->getName() << "\" cannot access input pass \""<< currentSecondaryPass->getName() <<"\". Please make sure you declared this input pass first in the scn file." << std::endl;
                    return;
                }
                //std::cout<<"***"<< this->getName() <<"*** GL_TEXTURE"<<nbFbo<<" color from "<<currentSecondaryPass->getName()<<std::endl;
                glActiveTexture(GL_TEXTURE0+nbFbo);
                glEnable(GL_TEXTURE_2D);
                glBindTexture(GL_TEXTURE_2D, currentSecondaryPass->getFBO()->getColorTexture());

                ++nbFbo;
            }
        }
        else
        {
            if(currentPass && (this->getName()!=currentPass->getName()))
            {

                if(input_tags.getValue().includes(currentPass->getTags()))
                {
                    if (!currentPass->hasFilledFbo())
                    {
                        std::cerr << "Error: SecondaryPass \"" << this->getName() << "\" cannot access input pass \""<< currentPass->getName() <<"\". Please make sure you declared this input pass first in the scn file." << std::endl;
                        return;
                    }

                    //std::cout<<"***"<< this->getName() <<"*** GL_TEXTURE"<<nbFbo<<" color from "<<currentPass->getName()<<std::endl;
                    glActiveTexture(GL_TEXTURE0+nbFbo);
                    glEnable(GL_TEXTURE_2D);
                    glBindTexture(GL_TEXTURE_2D, currentPass->getFBO()->getColorTexture());
                    ++nbFbo;

                    //std::cout<<"***"<< this->getName() <<"*** GL_TEXTURE"<<nbFbo<<" depth from "<<currentPass->getName()<<std::endl;
                    glActiveTexture(GL_TEXTURE0+nbFbo);
                    glEnable(GL_TEXTURE_2D);
                    glBindTexture(GL_TEXTURE_2D, currentPass->getFBO()->getDepthTexture());
                    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_LUMINANCE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_NONE);
                    ++nbFbo;
                }
            }
        }
    }
    glActiveTexture(GL_TEXTURE0);
}

void VisualManagerSecondaryPass::unbindInput()
{
    for(int j=0; j<nbFbo; ++j)
    {
        glActiveTexture(GL_TEXTURE0+j);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glActiveTexture(GL_TEXTURE0);
}


bool VisualManagerSecondaryPass::drawScene(VisualParams* vp)
{
    if(!multiPassEnabled)
        return false;

    if(renderToScreen.getValue())
    {
        shader_postproc->setFloat(0,"zNear", (float)vp->zNear());
        shader_postproc->setFloat(0,"zFar", (float)vp->zFar());
        shader_postproc->setInt(0,"width", vp->viewport()[2]);
        shader_postproc->setInt(0,"height", vp->viewport()[3]);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);

        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        glViewport(vp->viewport()[0],vp->viewport()[1],vp->viewport()[2],vp->viewport()[3]);

        bindInput(vp);//bind input textures from FBOs
        shader_postproc->start();
        traceFullScreenQuad();
        shader_postproc->stop();
        unbindInput();//unbind input textures

        glEnable(GL_LIGHTING);
        glEnable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        return true;
    }
    else
        return false;
}

} //namespaces
}
}
