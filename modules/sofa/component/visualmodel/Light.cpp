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
// C++ Implementation: Light
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <sofa/component/visualmodel/Light.h>
#include <sofa/component/visualmodel/LightManager.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(Light)

SOFA_DECL_CLASS(DirectionalLight)
//Register DirectionalLight in the Object Factory
int DirectionalLightClass = core::RegisterObject("Directional Light")
        .add< DirectionalLight >()
        ;

SOFA_DECL_CLASS(PositionalLight)
//Register PositionalLight in the Object Factory
int PositionalLightClass = core::RegisterObject("Positional Light")
        .add< PositionalLight >()
        ;

SOFA_DECL_CLASS(SpotLight)
//Register SpotLight in the Object Factory
int SpotLightClass = core::RegisterObject("Spot Light")
        .add< SpotLight >()
        ;

Light::Light()
    : lightID(0), shadowTexWidth(0),shadowTexHeight(0)
    , color(initData(&color, (Vector3) Vector3(1,1,1), "color", "Set the color of the light"))
    , zNear(initData(&zNear, (float) 4.0, "zNear", "Set minimum distance for view field"))
    , zFar(initData(&zFar, (float) 50.0, "zFar", "Set minimum distance for view field"))
    , shadowTextureSize (initData(&shadowTextureSize, (GLuint) 0, "shadowTextureSize", "Set size for shadow texture "))
    , enableShadow(initData(&enableShadow, (bool) true, "enableShadow", "Enable Shadow from this light"))
{

}

Light::~Light()
{
}

void Light::setID(const GLint& id)
{
    lightID = id;
}

void Light::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    LightManager* lm = context->core::objectmodel::BaseContext::get<LightManager>();

    lm->putLight(this);

}

void Light::initVisual()
{
    //Init Light part
    glLightf(GL_LIGHT0+lightID, GL_SPOT_CUTOFF, 180.0);
    GLfloat c[4] = { (GLfloat) color.getValue()[0], (GLfloat)color.getValue()[1], (GLfloat)color.getValue()[2], 1.0 };
    glLightfv(GL_LIGHT0+lightID, GL_AMBIENT, c);
    glLightfv(GL_LIGHT0+lightID, GL_DIFFUSE, c);
    glLightfv(GL_LIGHT0+lightID, GL_SPECULAR, c);
    glLightf(GL_LIGHT0+lightID, GL_LINEAR_ATTENUATION, 0.0);

    //init Shadow part
    computeShadowMapSize();
    //Shadow part
    //Shadow texture init
    shadowFBO.init(shadowTexWidth, shadowTexHeight);

}

void Light::reinit()
{

    initVisual();

}

void Light::drawLight()
{

}

void Light::preDrawShadow()
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    shadowFBO.start();
}

void Light::postDrawShadow()
{
    //Unbind fbo
    shadowFBO.stop();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

void Light::computeShadowMapSize()
{
    // Current viewport
    GLint		viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    GLint windowWidth = viewport[2];
    GLint windowHeight = viewport[3];

    if (shadowTextureSize.getValue() <= 0)
    {
        //Get the size of the shadow map
        if (windowWidth >= 1024 && windowHeight >= 1024)
        {
            shadowTexWidth = shadowTexHeight = 1024;
        }
        else if (windowWidth >= 512 && windowHeight >= 512)
        {
            shadowTexWidth = shadowTexHeight = 512;
        }
        else if (windowWidth >= 256 && windowHeight >= 256)
        {
            shadowTexWidth = shadowTexHeight = 256;
        }
        else
        {
            shadowTexWidth = shadowTexHeight = 128;
        }
    }
    else
        shadowTexWidth = shadowTexHeight = shadowTextureSize.getValue();
}


GLuint Light::getShadowMapSize()
{
    return shadowTexWidth;
}


DirectionalLight::DirectionalLight():
    direction(initData(&direction, (Vector3) Vector3(0,0,-1), "direction", "Set the direction of the light"))
{

}

DirectionalLight::~DirectionalLight()
{

}

void DirectionalLight::initVisual()
{
    Light::initVisual();

}

void DirectionalLight::reinit()
{
    initVisual();
}

void DirectionalLight::drawLight()
{
    Light::drawLight();
    GLfloat dir[4];

    dir[0]=(GLfloat)(direction.getValue()[0]);
    dir[1]=(GLfloat)(direction.getValue()[1]);
    dir[2]=(GLfloat)(direction.getValue()[2]);
    dir[3]=0.0; // directional

    glLightfv(GL_LIGHT0+lightID, GL_POSITION, dir);
}

PositionalLight::PositionalLight():
    position(initData(&position, (Vector3) Vector3(-0.7,0.3,0.0), "position", "Set the position of the light")),
    attenuation(initData(&attenuation, (float) 0.0, "attenuation", "Set the attenuation of the light"))
{

}

PositionalLight::~PositionalLight()
{

}

void PositionalLight::initVisual()
{
    Light::initVisual();

}

void PositionalLight::reinit()
{
    initVisual();

}

void PositionalLight::drawLight()
{
    Light::drawLight();

    GLfloat pos[4];
    pos[0]=(GLfloat)(position.getValue()[0]);
    pos[1]=(GLfloat)(position.getValue()[1]);
    pos[2]=(GLfloat)(position.getValue()[2]);
    pos[3]=1.0; // positional
    glLightfv(GL_LIGHT0+lightID, GL_POSITION, pos);

    glLightf(GL_LIGHT0+lightID, GL_LINEAR_ATTENUATION, attenuation.getValue());

}


SpotLight::SpotLight():
    direction(initData(&direction, (Vector3) Vector3(0,0,-1), "direction", "Set the direction of the light")),
    cutoff(initData(&cutoff, (float) 30.0, "cutoff", "Set the angle (cutoff) of the spot")),
    exponent(initData(&exponent, (float) 20.0, "exponent", "Set the exponent of the spot"))
{

}

SpotLight::~SpotLight()
{

}

void SpotLight::initVisual()
{
    PositionalLight::initVisual();

}

void SpotLight::reinit()
{
    initVisual();

}

void SpotLight::drawLight()
{
    PositionalLight::drawLight();

    GLfloat dir[]= {(GLfloat)(direction.getValue()[0]), (GLfloat)(direction.getValue()[1]), (GLfloat)(direction.getValue()[2])};
    glLightf(GL_LIGHT0+lightID, GL_SPOT_CUTOFF, cutoff.getValue());
    glLightfv(GL_LIGHT0+lightID, GL_SPOT_DIRECTION, dir);
    glLightf(GL_LIGHT0+lightID, GL_SPOT_EXPONENT, exponent.getValue());

}

void SpotLight::preDrawShadow()
{
    Light::preDrawShadow();

    //float d = 4.0 * tan(cutoff.getValue()*3.14159/180);

    //Projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //glFrustum(-d, d, -d, d, ZNEAR, ZFAR);
    gluPerspective(2*cutoff.getValue(),1.0, zNear.getValue(), zFar.getValue());

    //Modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
//	gluLookAt(position.getValue()[0], position.getValue()[1], position.getValue()[2],
//			position.getValue()[0] + direction.getValue()[0],
//			position.getValue()[1] + direction.getValue()[1],
//			position.getValue()[2] + direction.getValue()[2],
//			0,1,0);
    gluLookAt(position.getValue()[0], position.getValue()[1], position.getValue()[2],0 ,0 ,0, direction.getValue()[0], direction.getValue()[1], direction.getValue()[2]);

    //Save the two matrices
    glGetFloatv(GL_PROJECTION_MATRIX, lightMatProj);
    glGetFloatv(GL_MODELVIEW_MATRIX, lightMatModelview);

    //glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, shadowFBO);

    glViewport(0, 0, shadowTexWidth, shadowTexHeight);

    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
}

GLuint SpotLight::getShadowTexture()
{
    //return debugVisualShadowTexture;
    //return shadowTexture;
    return shadowFBO.getDepthTexture();
}

GLfloat* SpotLight::getProjectionMatrix()
{
    return lightMatProj;
}

GLfloat* SpotLight::getModelviewMatrix()
{
    return lightMatModelview;
}

}

} //namespace component

} //namespace sofa
