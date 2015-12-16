/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaOpenglVisual/LightManager.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

#ifdef SOFA_HAVE_GLEW
#include <SofaOpenglVisual/OglTexture.h>
#endif // SOFA_HAVE_GLEW

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace helper::gl;
using namespace simulation;
using namespace core::visual;

SOFA_DECL_CLASS(LightManager)
//Register LightManager in the Object Factory
int LightManagerClass = core::RegisterObject("LightManager")
        .add< LightManager >()
        ;

LightManager::LightManager()
    : shadowsEnabled(initData(&shadowsEnabled, (bool) false, "shadows", "Enable Shadow in the scene"))
    , softShadowsEnabled(initData(&softShadowsEnabled, (bool) false, "softShadows", "If Shadows enabled, Enable Variance Soft Shadow in the scene"))
    , ambient(initData(&ambient, defaulttype::Vec4f(0.0f,0.0f,0.0f,0.0f), "ambient", "Ambient lights contribution"))
{
    //listen by default, in order to get the keys to activate shadows
    if(!f_listening.isSet())
        f_listening.setValue(true);
}

LightManager::~LightManager()
{
    //restoreDefaultLight();
}

void LightManager::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
#ifdef SOFA_HAVE_GLEW
    context->get<sofa::component::visualmodel::OglShadowShader, sofa::helper::vector<sofa::component::visualmodel::OglShadowShader::SPtr> >(&shadowShaders, core::objectmodel::BaseContext::SearchRoot);

    if (shadowShaders.empty() && shadowsEnabled.getValue())
    {
        serr << "LightManager: No OglShadowShaders found ; shadow will be disabled."<< sendl;
        shadowsEnabled.setValue(false);
        //return;
    }

    for(unsigned int i=0 ; i<shadowShaders.size() ; ++i)
    {
        shadowShaders[i]->initShaders((unsigned int)lights.size(), softShadowsEnabled.getValue());
        shadowShaders[i]->setCurrentIndex(shadowsEnabled.getValue() ? 1 : 0);
    }
#endif
    lightModelViewMatrix.resize(lights.size());

}

void LightManager::bwdInit()
{
#ifdef SOFA_HAVE_GLEW
    for(unsigned int i=0 ; i<shadowShaders.size() ; ++i)
    {
        shadowShaders[i]->setCurrentIndex(shadowsEnabled.getValue() ? 1 : 0);
    }
#endif
}


void LightManager::initVisual()
{
#ifdef SOFA_HAVE_GLEW
    for(unsigned int i=0 ; i<shadowShaders.size() ; ++i)
        shadowShaders[i]->initVisual();

    ///TODO: keep trace of all active textures at the same time, with a static factory
    ///or something like that to avoid conflics with color texture declared in the scene file.
    //Check unit textures availability
    sofa::helper::vector<sofa::component::visualmodel::OglTexture::SPtr> sceneTextures;
    this->getContext()->get<sofa::component::visualmodel::OglTexture, sofa::helper::vector<sofa::component::visualmodel::OglTexture::SPtr> >(&sceneTextures, core::objectmodel::BaseContext::SearchRoot);

    GLint maxTextureUnits;
    glGetIntegerv(GL_MAX_TEXTURE_UNITS, &maxTextureUnits);
    std::vector<bool> availableUnitTextures;
    availableUnitTextures.resize(maxTextureUnits);
    std::fill(availableUnitTextures.begin(), availableUnitTextures.end(), true);
    for(unsigned int i=0 ; i<sceneTextures.size() ; i++)
    {
        availableUnitTextures[sceneTextures[i]->getTextureUnit()] = false;
    }

    for (std::vector<Light::SPtr>::iterator itl = lights.begin(); itl != lights.end() ; ++itl)
    {
        (*itl)->initVisual();
        unsigned short shadowTextureUnit = (*itl)->getShadowTextureUnit();
        //if given unit is available and correct
        if(shadowTextureUnit < maxTextureUnits &&
                availableUnitTextures[shadowTextureUnit] == true)
        {
            availableUnitTextures[shadowTextureUnit] = false;
        }
        else //otherwise search the first one available
        {
            bool found = false;
            for(unsigned short i=0 ; i < availableUnitTextures.size() && !found; i++)
            {
                found = availableUnitTextures[i];
                if(found)
                {
                    (*itl)->setShadowTextureUnit(i);
                    availableUnitTextures[i] = false;
                }

            }

        }
    }

#endif // SOFA_HAVE_GLEW

}

void LightManager::putLight(Light::SPtr light)
{
    if (lights.size() >= MAX_NUMBER_OF_LIGHTS)
    {
        serr << "The maximum of lights permitted ( "<< MAX_NUMBER_OF_LIGHTS << " ) has been reached." << sendl;
        return ;
    }

    light->setID((GLint)lights.size());
    lights.push_back(light) ;
}

void LightManager::putLights(std::vector<Light::SPtr> lights)
{
    for (std::vector<Light::SPtr>::iterator itl = lights.begin(); itl != lights.end() ; ++itl)
        putLight(*itl);
}

void LightManager::makeShadowMatrix(unsigned int i)
{
    const GLfloat* lp = lights[i]->getProjectionMatrix();
    const GLfloat* lmv = lights[i]->getModelviewMatrix();

    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glTranslatef(0.5f, 0.5f, 0.5f +( -0.006f) );
    glScalef(0.5f, 0.5f, 0.5f);

    glMultMatrixf(lp); // now multiply by the matrices we have retrieved before
    glMultMatrixf(lmv);
    sofa::defaulttype::Mat<4,4,float> model2;
    glGetFloatv(GL_MODELVIEW_MATRIX,model2.ptr());
    model2.invert(model2);

    glMultMatrixf(model2.ptr());

    if (lightModelViewMatrix.size() > 0)
    {
        lightModelViewMatrix[i] = lmv;
        lightProjectionMatrix[i] = lp;
    }
    else
    {
        lightModelViewMatrix.resize(lights.size());
        lightProjectionMatrix.resize(lights.size());
        lightModelViewMatrix[i] = lmv;
        lightProjectionMatrix[i] = lp;
    }
//    std::cout << "lightModelViewMatrix[i] "<<i << " -> " << lightModelViewMatrix[i] << std::endl;
//    std::cout << "lightProjectionMatrix[i] "<<i << " -> " << lightProjectionMatrix[i] << std::endl;

    glMatrixMode(GL_MODELVIEW);
}

void LightManager::fwdDraw(core::visual::VisualParams* vp)
{

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient.getValue().ptr());
    unsigned int id = 0;
    for (std::vector<Light::SPtr>::iterator itl = lights.begin(); itl != lights.end() ; ++itl)
    {
        glEnable(GL_LIGHT0+id);
        (*itl)->drawLight();
        ++id;
    }

#ifdef SOFA_HAVE_GLEW
    const core::visual::VisualParams::Pass pass = vp->pass();
    GLint lightFlag[MAX_NUMBER_OF_LIGHTS];
    GLint shadowTextureID[MAX_NUMBER_OF_LIGHTS];
    //GLfloat lightModelViewProjectionMatrices[MAX_NUMBER_OF_LIGHTS*16];
    GLfloat zNears[MAX_NUMBER_OF_LIGHTS];
    GLfloat zFars[MAX_NUMBER_OF_LIGHTS];

    if(pass != core::visual::VisualParams::Shadow)
    {
        if (!shadowShaders.empty())
        {
            glEnable(GL_LIGHTING);
            for (unsigned int i=0 ; i < lights.size() ; ++i)
            {
                unsigned short shadowTextureUnit = lights[i]->getShadowTextureUnit();
                glActiveTexture(GL_TEXTURE0+shadowTextureUnit);
                glEnable(GL_TEXTURE_2D);

                if (softShadowsEnabled.getValue())
                    glBindTexture(GL_TEXTURE_2D, lights[i]->getColorTexture());
                else
                    //glBindTexture(GL_TEXTURE_2D, lights[i]->getDepthTexture());
                    glBindTexture(GL_TEXTURE_2D, lights[i]->getColorTexture());

                lightFlag[i] = 1;
                shadowTextureID[i] = 0;
                zNears[i] = (GLfloat) lights[i]->p_zNear.getValue();
                zFars[i] = (GLfloat) lights[i]->p_zFar.getValue();

                if (shadowsEnabled.getValue() && lights[i]->shadowsEnabled.getValue())
                {
                    lightFlag[i] = 2;
                    shadowTextureID[i] = shadowTextureUnit;
                }

                glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
                glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
                glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB);
                glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_LUMINANCE);
                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

                makeShadowMatrix(i);
            }

            for (unsigned int i = (unsigned int)lights.size() ; i< MAX_NUMBER_OF_LIGHTS ; i++)
            {
                lightFlag[i] = 0;
                shadowTextureID[i] = 0;

                /*for(unsigned int j=0 ; j<4; j++)
                    for(unsigned int k=0 ; k<4; k++)
                        lightModelViewProjectionMatrices[16*i+j*4+k] = 0.0;*/
            }

            for(unsigned int i=0 ; i<shadowShaders.size() ; ++i)
            {

                shadowShaders[i]->setIntVector(shadowShaders[i]->getCurrentIndex() , "lightFlag" , MAX_NUMBER_OF_LIGHTS, lightFlag);
                shadowShaders[i]->setIntVector(shadowShaders[i]->getCurrentIndex() , "shadowTexture" , MAX_NUMBER_OF_LIGHTS, shadowTextureID);
                shadowShaders[i]->setIntVector(shadowShaders[i]->getCurrentIndex() , "shadowTextureUnit" , MAX_NUMBER_OF_LIGHTS, shadowTextureID);
                shadowShaders[i]->setFloatVector(shadowShaders[i]->getCurrentIndex() , "zNear" , MAX_NUMBER_OF_LIGHTS, zNears);
                shadowShaders[i]->setFloatVector(shadowShaders[i]->getCurrentIndex() , "zFar" , MAX_NUMBER_OF_LIGHTS, zFars);
                //shadowShaders[i]->setFloatVector4(shadowShaders[i]->getCurrentIndex() , "lightPosition" , MAX_NUMBER_OF_LIGHTS,  &(lightPosition[0][0]));
                //shadowShaders[i]->setMatrix4(shadowShaders[i]->getCurrentIndex() , "lightModelViewMatrix" , MAX_NUMBER_OF_LIGHTS, false, (lightModelViewMatrix[0].ptr()));
                //shadowShaders[i]->setMatrix4(shadowShaders[i]->getCurrentIndex() , "lightProjectionMatrix" , MAX_NUMBER_OF_LIGHTS, false, (lightModelViewMatrix[0].ptr()));
                //shadowShader->start();
            }

        }
    }

    glActiveTexture(GL_TEXTURE0);
#endif
}

void LightManager::bwdDraw(core::visual::VisualParams* )
{
#ifdef SOFA_HAVE_GLEW
    for(unsigned int i=0 ; i<lights.size() ; ++i)
    {
        unsigned short shadowTextureUnit = lights[i]->getShadowTextureUnit();
        glActiveTexture(GL_TEXTURE0+shadowTextureUnit);
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
    }

    glActiveTexture(GL_TEXTURE0);
#endif // SOFA_HAVE_GLEW
    for (unsigned int i=0 ; i<MAX_NUMBER_OF_LIGHTS ; ++i)
        glDisable(GL_LIGHT0+i);

    //reset Texture Matrix
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);


}

void LightManager::draw(const core::visual::VisualParams* )
{
    /*
    //Debug
    //reset Texture Matrix
    glMatrixMode(GL_TEXTURE);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    for(unsigned int i=0 ; i < lights.size() ; i++)
    {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, lights[i]->getDepthTexture());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_NONE);
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_LUMINANCE);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glBegin(GL_QUADS);
    glColor3f(1.0,0.0,0.0) ; glTexCoord2f(0,0); glVertex3f(0 + i*20, 20, -10);
    glColor3f(0.0,1.0,0.0) ; glTexCoord2f(1,0); glVertex3f(0 + i*20, 40, -10);
    glColor3f(0.0,0.0,1.0) ; glTexCoord2f(1,1); glVertex3f(20 + i*20, 40, -10);
    glColor3f(0.0,0.0,0.0) ; glTexCoord2f(0,1); glVertex3f(20 + i*20, 20, -10);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, lights[i]->getColorTexture());

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glBegin(GL_QUADS);
    glColor3f(1.0,0.0,0.0) ; glTexCoord2f(0,0); glVertex3f(40 + i*20, 20, -10);
    glColor3f(0.0,1.0,0.0) ; glTexCoord2f(1,0); glVertex3f(40 + i*20, 40, -10);
    glColor3f(0.0,0.0,1.0) ; glTexCoord2f(1,1); glVertex3f(60 + i*20, 40, -10);
    glColor3f(0.0,0.0,0.0) ; glTexCoord2f(0,1); glVertex3f(60 + i*20, 20, -10);
    glEnd();
    }

    glMatrixMode(GL_TEXTURE);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glEnable(GL_LIGHTING);
    glBindTexture(GL_TEXTURE_2D, 0);
    */
}

void LightManager::clear()
{
    for (unsigned int i=0 ; i<MAX_NUMBER_OF_LIGHTS ; ++i)
        glDisable(GL_LIGHT0+i);
    lights.clear();
}

void LightManager::reinit()
{
    for (std::vector<Light::SPtr>::iterator itl = lights.begin(); itl != lights.end() ; ++itl)
    {
        (*itl)->reinit();
    }
}

void LightManager::preDrawScene(VisualParams* vp)
{
#ifdef SOFA_HAVE_GLEW
	if(shadowsEnabled.getValue())
	{
        for (std::vector<Light::SPtr>::iterator itl = lights.begin(); itl != lights.end() ; ++itl)
		{
			(*itl)->preDrawShadow(vp);
			vp->pass() = core::visual::VisualParams::Shadow;
			simulation::VisualDrawVisitor vdv(vp);

			vdv.execute ( getContext() );

			(*itl)->postDrawShadow();
		}
		const core::visual::VisualParams::Viewport& viewport = vp->viewport();
		//restore viewport
		glViewport(viewport[0], viewport[1], viewport[2] , viewport[3]);
	}
#endif
}

bool LightManager::drawScene(VisualParams* /*vp*/)
{
    return false;
}

void LightManager::postDrawScene(VisualParams* vp)
{
    restoreDefaultLight(vp);
}

void LightManager::restoreDefaultLight(VisualParams* vp)
{
    //restore default light
    GLfloat	ambientLight[4];
    GLfloat	diffuseLight[4];
    GLfloat	specular[4];
    GLfloat	lightPosition[4];

    lightPosition[0] = -0.7f;
    lightPosition[1] = 0.3f;
    lightPosition[2] = 0.0f;
    lightPosition[3] = 1.0f;

    ambientLight[0] = 0.5f;
    ambientLight[1] = 0.5f;
    ambientLight[2] = 0.5f;
    ambientLight[3] = 1.0f;

    diffuseLight[0] = 0.9f;
    diffuseLight[1] = 0.9f;
    diffuseLight[2] = 0.9f;
    diffuseLight[3] = 1.0f;

    specular[0] = 1.0f;
    specular[1] = 1.0f;
    specular[2] = 1.0f;
    specular[3] = 1.0f;

    // Setup 'light 0'
    // It crashes here in batch mode on Mac... probably the lack of GL context ?

    if (vp->isSupported(core::visual::API_OpenGL))
    {
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
        glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
        glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, 180);

        glEnable(GL_LIGHT0);
    }
}

void LightManager::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent *ev = static_cast<sofa::core::objectmodel::KeypressedEvent *>(event);
        switch(ev->getKey())
        {

        case 'l':
        case 'L':
#ifdef SOFA_HAVE_GLEW
            if (!shadowShaders.empty())
            {
                bool b = shadowsEnabled.getValue();
                shadowsEnabled.setValue(!b);
                if (!shadowShaders.empty())
                {
                    for (unsigned int i=0 ; i < shadowShaders.size() ; ++i)
                    {
                        shadowShaders[i]->setCurrentIndex(shadowsEnabled.getValue() ? 1 : 0);
                        shadowShaders[i]->updateVisual();
                    }
                    for (std::vector<Light::SPtr>::iterator itl = lights.begin(); itl != lights.end() ; ++itl)
                    {
                        (*itl)->updateVisual();
                    }
                    this->updateVisual();
                }

                sout << "Shadows : "<<(shadowsEnabled.getValue()?"ENABLED":"DISABLED")<<sendl;
            }
#endif
            break;
        }
    }

}

}//namespace visualmodel

}//namespace component

}//namespace sofa
