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
 * PostProcessManager.cpp
 *
 *  Created on: 12 janv. 2009
 *      Author: froy
 */

#include "PostProcessManager.h"
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(PostProcessManager)
//Register PostProcessManager in the Object Factory
int PostProcessManagerClass = core::RegisterObject("PostProcessManager")
        .add< PostProcessManager >()
        ;

using namespace core::visual;

const std::string PostProcessManager::DEPTH_OF_FIELD_VERTEX_SHADER = "shaders/depthOfField.vert";
const std::string PostProcessManager::DEPTH_OF_FIELD_FRAGMENT_SHADER = "shaders/depthOfField.frag";

PostProcessManager::PostProcessManager()
    :zNear(initData(&zNear, (double) 1.0, "zNear", "Set zNear distance (for Depth Buffer)"))
    ,zFar(initData(&zFar, (double) 100.0, "zFar", "Set zFar distance (for Depth Buffer)"))
    ,postProcessEnabled (true)
{
    // TODO Auto-generated constructor stub

}

PostProcessManager::~PostProcessManager()
{

}



void PostProcessManager::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    dofShader = context->core::objectmodel::BaseContext::get<sofa::component::visualmodel::OglShader>();

    if (!dofShader)
    {
        serr << "PostProcessingManager: OglShader not found ; no post process applied."<< sendl;
        postProcessEnabled = false;
        return;
    }
}

void PostProcessManager::initVisual()
{
    if (postProcessEnabled)
    {
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        GLint windowWidth = viewport[2];
        GLint windowHeight = viewport[3];

        fbo = std::unique_ptr<helper::gl::FrameBufferObject>(new helper::gl::FrameBufferObject());
        fbo->init(windowWidth, windowHeight);


        /*dofShader = new OglShader();
        dofShader->vertFilename.setValue(vertFilename.getValue());
        dofShader->fragFilename.setValue(fragFilename.getValue());

        dofShader->init();
        dofShader->initVisual();

        */
        dofShader->setInt(0, "colorTexture", 0);
        dofShader->setInt(0, "depthTexture", 1);
    }
}

void PostProcessManager::preDrawScene(VisualParams* vp)
{
    const VisualParams::Viewport& viewport = vp->viewport();

    if (postProcessEnabled)
    {
        fbo->setSize(viewport[2], viewport[3]);
        fbo->start();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluPerspective(60.0,1.0, zNear.getValue(), zFar.getValue());

        glMatrixMode(GL_MODELVIEW);
        vp->pass() = VisualParams::Std;
        simulation::VisualDrawVisitor vdv( vp);
        vdv.execute ( getContext() );
        vp->pass() = VisualParams::Transparent;
        simulation::VisualDrawVisitor vdvt( vp );
        vdvt.execute ( getContext() );

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);

        gluPerspective(60.0,1.0, vp->zNear(), vp->zFar());
        glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);

        fbo->stop();
    }
}

bool PostProcessManager::drawScene(VisualParams* vp)
{
    const VisualParams::Viewport& viewport = vp->viewport();
    if (postProcessEnabled)
    {
        float vxmax, vymax;
        float vxmin, vymin;
        float txmax,tymax;
        float txmin,tymin;

        txmin = tymin = 0.0;
        vxmin = vymin = -1.0;
        vxmax = vymax = txmax = tymax = 1.0;

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);

        glActiveTexture(GL_TEXTURE0);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, fbo->getColorTexture());

        glActiveTexture(GL_TEXTURE1);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, fbo->getDepthTexture());
        glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_LUMINANCE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_NONE);

        float pixelSize[2];
        pixelSize[0] = (float)1.0/viewport[2];
        pixelSize[1] = (float)1.0/viewport[3];

        //dofShader->setInt(0, "colorTexture", 0);
        //dofShader->setInt(0, "depthTexture", 1);
        dofShader->setFloat2(0, "pixelSize", pixelSize[0], pixelSize[1]);

        dofShader->start();

        glBegin(GL_QUADS);
        {
            glTexCoord3f(txmin,tymax,0.0); glVertex3f(vxmin,vymax,0.0);
            glTexCoord3f(txmax,tymax,0.0); glVertex3f(vxmax,vymax,0.0);
            glTexCoord3f(txmax,tymin,0.0); glVertex3f(vxmax,vymin,0.0);
            glTexCoord3f(txmin,tymin,0.0); glVertex3f(vxmin,vymin,0.0);
        }
        glEnd();

        dofShader->stop();
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);

        glEnable(GL_LIGHTING);
        glEnable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        return true;
    }

    return false;
}

void PostProcessManager::postDrawScene(VisualParams* /*vp*/)
{

}

void PostProcessManager::handleEvent(sofa::core::objectmodel::Event* /*event*/)
{
}

} //visualmodel

} //component

} //sofa
