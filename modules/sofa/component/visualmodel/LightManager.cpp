/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
//
// C++ Implementation: LightManager
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/component/visualmodel/LightManager.h>
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace helper::gl;
using namespace simulation;

SOFA_DECL_CLASS(LightManager)
//Register LightManager in the Object Factory
int LightManagerClass = core::RegisterObject("LightManager")
        .add< LightManager >()
        ;

LightManager::LightManager()
    :shadowEnabled(false)
    ,debugViewDepthBuffer(initData(&debugViewDepthBuffer, (bool) false, "debugViewDepthBuffer", "DEBUG : View the buffer depth as seen by the light(s)"))
{

}

LightManager::~LightManager()
{

}

void LightManager::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    shadowShader = context->core::objectmodel::BaseContext::get<sofa::component::visualmodel::OglShadowShader>();

    if (!shadowShader)
    {
        std::cerr << "LightManager: OglShadowShader not found ; shadow will be disabled."<< std::endl;
        shadowEnabled = false;
        return;
    }
}


void LightManager::initVisual()
{
    if (shadowShader)
    {
        shadowShader->initShaders(lights.size());
        shadowShader->initVisual();
    }

    for (std::vector<Light*>::iterator itl = lights.begin(); itl != lights.end() ; itl++)
    {
        (*itl)->initVisual();
    }
}

void LightManager::putLight(Light* light)
{
    if (lights.size() >= MAX_NUMBER_OF_LIGHTS)
    {
        serr << "The maximum of lights permitted ( "<< MAX_NUMBER_OF_LIGHTS << " ) has been reached." << sendl;
        return ;
    }

    light->setID(lights.size());
    lights.push_back(light) ;
}

void LightManager::putLights(std::vector<Light*> lights)
{
    for (std::vector<Light*>::iterator itl = lights.begin(); itl != lights.end() ; itl++)
        putLight(*itl);
}

void LightManager::makeShadowMatrix(unsigned int i)
{
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glTranslatef(0.5f, 0.5f, 0.5f +( -0.006f) );
    glScalef(0.5f, 0.5f, 0.5f);

    glMultMatrixf(lights[i]->getProjectionMatrix()); // now multiply by the matrices we have retrieved before
    glMultMatrixf(lights[i]->getModelviewMatrix());
    sofa::defaulttype::Mat<4,4,float> model;
    glGetFloatv(GL_MODELVIEW_MATRIX,model.ptr());
    model.invert(model);

    glMultMatrixf(model.ptr());
    glMatrixMode(GL_MODELVIEW);
}

void LightManager::fwdDraw(Pass)
{
    if (shadowShader  && !debugViewDepthBuffer.getValue())
    {
        GLint* lightFlag = new GLint[MAX_NUMBER_OF_LIGHTS];
        GLint* shadowTextureID = new GLint [MAX_NUMBER_OF_LIGHTS];

        glEnable(GL_LIGHTING);
        for (unsigned int i=0 ; i < lights.size() ; i++)
        {
            glActiveTexture(GL_TEXTURE0 + i);
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, lights[i]->getShadowTexture());

            lightFlag[i] = 1;

            if (shadowEnabled && lights[i]->enableShadow.getValue())
            {
                lightFlag[i] = 2;
                shadowTextureID[i] = i;
            }

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB);
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

            makeShadowMatrix(i);
        }
        for (unsigned int i = lights.size() ; i< MAX_NUMBER_OF_LIGHTS ; i++)
        {
            lightFlag[i] = 0;
            shadowTextureID[i] = -1;
        }

        shadowShader->setIntVector(shadowShader->getCurrentIndex() , "lightFlag" , MAX_NUMBER_OF_LIGHTS, lightFlag);
        shadowShader->setIntVector(shadowShader->getCurrentIndex() , "shadowTexture" , MAX_NUMBER_OF_LIGHTS, shadowTextureID);

        shadowShader->start();

        delete lightFlag;
        delete shadowTextureID;

    }

}
void LightManager::bwdDraw(Pass)
{
    if (shadowShader && shadowEnabled)
    {
        shadowShader->stop();
    }
}

void LightManager::draw()
{

    unsigned int id = 0;
    for (std::vector<Light*>::iterator itl = lights.begin(); itl != lights.end() ; itl++)
    {
        glEnable(GL_LIGHT0+id);
        (*itl)->drawLight();
        id++;
    }
    /*
    #ifdef SOFA_DEV_SHADOW
    	if (shadowShader && debugViewDepthBuffer.getValue())
    	{
    		//shadowShader.stop();

    		GLint		viewport[4];
    		glGetIntegerv(GL_VIEWPORT, viewport);

    		//SHOW Depth texture

    		glClearColor(0.0f, 0.0f, 0.2f, 0.5f);
    		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear Screen And Depth Buffer
    		glDisable(GL_LIGHTING);
    		glEnable(GL_TEXTURE_2D);

    		glColor4f(1.0f,1.0f,1.0f,1.0f);
    		for (unsigned int i=0 ; i < lights.size() ; i++)
    		{
    			glBindTexture(GL_TEXTURE_2D, lights[i]->getShadowTexture());
    #if defined(GL_ARB_shadow)
    			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_NONE);
    			glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_LUMINANCE);
    #endif

    			glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    			glPushMatrix();
    			glTranslatef(i,0.0,0.0);
    			glBegin(GL_QUADS);
    			// Front Face
    			glNormal3f( 0.0f, 0.0f, 1.0);
    			glTexCoord2f(0.0f, 0.0f); glVertex3f(-0.5f, -0.5,  0.0);
    			glTexCoord2f(1.0f, 0.0f); glVertex3f( 0.5, -0.5,  0.0);
    			glTexCoord2f(1.0f, 1.0f); glVertex3f( 0.5,  0.5,  0.0);
    			glTexCoord2f(0.0f, 1.0f); glVertex3f(-0.5,  0.5,  0.0);
    			glEnd();
    			glPopMatrix();

    		}
    		glBindTexture(GL_TEXTURE_2D, 0);
    	    glEnable(GL_DEPTH_TEST);
    	    glEnable(GL_LIGHTING);

    	    //shadowShader.start();
    	}
    #endif
    */
}

void LightManager::clear()
{
    for (unsigned int i=0 ; i<MAX_NUMBER_OF_LIGHTS ; i++)
        glDisable(GL_LIGHT0+i);
    lights.clear();
}

void LightManager::reinit()
{
    for (std::vector<Light*>::iterator itl = lights.begin(); itl != lights.end() ; itl++)
    {
        (*itl)->reinit();
    }
}

void LightManager::preDrawScene(VisualParameters* vp)
{
    for (std::vector<Light*>::iterator itl = lights.begin(); itl != lights.end() ; itl++)
    {
        if(shadowEnabled)
        {
            (*itl)->preDrawShadow();

            simulation::VisualDrawVisitor vdv( core::VisualModel::Std );
            vdv.execute ( getContext() );
        }
    }

    for (std::vector<Light*>::iterator itl = lights.begin(); itl != lights.end() ; itl++)
    {
        if(shadowEnabled)
        {
            (*itl)->postDrawShadow();
        }
    }
    //restore viewport
    glViewport(0, 0, vp->viewport[2] , vp->viewport[3]);
}

bool LightManager::drawScene(VisualParameters* /*vp*/)
{
    return false;
}

void LightManager::postDrawScene(VisualParameters* /*vp*/)
{

}

void LightManager::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        switch(ev->getKey())
        {

        case 'l':
        case 'L':
            if (shadowShader)
            {
                shadowEnabled = !shadowEnabled;
                std::cout << "Shadows : "<<(shadowEnabled?"ENABLED":"DISABLED")<<std::endl;
            }
            break;
        }
    }

}

}//namespace visualmodel

}//namespace component

}//namespace sofa
