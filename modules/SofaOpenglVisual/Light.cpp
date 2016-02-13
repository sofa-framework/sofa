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

#include <SofaOpenglVisual/Light.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaOpenglVisual/LightManager.h>
#include <sofa/helper/system/glu.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/Simulation.h>

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

using sofa::defaulttype::Vector3;

#ifdef SOFA_HAVE_GLEW
const std::string Light::PATH_TO_GENERATE_DEPTH_TEXTURE_VERTEX_SHADER = "shaders/softShadows/VSM/generate_depth_texture.vert";
const std::string Light::PATH_TO_GENERATE_DEPTH_TEXTURE_FRAGMENT_SHADER = "shaders/softShadows/VSM/generate_depth_texture.frag";

const std::string Light::PATH_TO_BLUR_TEXTURE_VERTEX_SHADER = "shaders/softShadows/VSM/blur_texture.vert";
const std::string Light::PATH_TO_BLUR_TEXTURE_FRAGMENT_SHADER = "shaders/softShadows/VSM/blur_texture.frag";
#endif

Light::Light()
    : lightID(0), shadowTexWidth(0),shadowTexHeight(0)
#ifdef SOFA_HAVE_GLEW
    , shadowFBO(true, true, true), blurHFBO(false,false,true), blurVFBO(false,false,true)
    , depthShader(sofa::core::objectmodel::New<OglShader>())
    , blurShader(sofa::core::objectmodel::New<OglShader>())
#endif
    , color(initData(&color, (Vector3) Vector3(1,1,1), "color", "Set the color of the light"))
    , shadowTextureSize (initData(&shadowTextureSize, (GLuint) 0, "shadowTextureSize", "Set size for shadow texture "))
    , drawSource(initData(&drawSource, (bool) false, "drawSource", "Draw Light Source"))
    , p_zNear(initData(&p_zNear, "zNear", "Camera's ZNear"))
    , p_zFar(initData(&p_zFar, "zFar", "Camera's ZFar"))
    , shadowsEnabled(initData(&shadowsEnabled, (bool) true, "shadowsEnabled", "Enable Shadow from this light"))
    , softShadows(initData(&softShadows, (bool) false, "softShadows", "Turn on Soft Shadow from this light"))
    , d_textureUnit(initData(&d_textureUnit, (unsigned short) 1, "textureUnit", "Texture unit for the genereated shadow texture"))
    , needUpdate(false)
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

    if(lm)
    {
        lm->putLight(this);
        softShadows.setParent(&(lm->softShadowsEnabled));
        //softShadows = lm->softShadowsEnabled.getValue();
    }
    else
    {
        serr << "No LightManager found" << sendl;
    }

}

void Light::initVisual()
{
    //init Shadow part
    computeShadowMapSize();
    //Shadow part
    //Shadow texture init
#ifdef SOFA_HAVE_GLEW
    shadowFBO.init(shadowTexWidth, shadowTexHeight);
    blurHFBO.init(shadowTexWidth, shadowTexHeight);
    blurVFBO.init(shadowTexWidth, shadowTexHeight);
    depthShader->vertFilename.setValueAsString(PATH_TO_GENERATE_DEPTH_TEXTURE_VERTEX_SHADER);
    depthShader->fragFilename.setValueAsString(PATH_TO_GENERATE_DEPTH_TEXTURE_FRAGMENT_SHADER);
    depthShader->init();
    depthShader->initVisual();
    blurShader->vertFilename.setValueAsString(PATH_TO_BLUR_TEXTURE_VERTEX_SHADER);
    blurShader->fragFilename.setValueAsString(PATH_TO_BLUR_TEXTURE_FRAGMENT_SHADER);
    blurShader->init();
    blurShader->initVisual();
#endif
}

void Light::updateVisual()
{
    if (!needUpdate) return;
    computeShadowMapSize();
    needUpdate = false;
}

void Light::reinit()
{
    needUpdate = true;
}

void Light::drawLight()
{
    if (needUpdate)
        updateVisual();
    glLightf(GL_LIGHT0+lightID, GL_SPOT_CUTOFF, 180.0);
    GLfloat c[4] = { (GLfloat) color.getValue()[0], (GLfloat)color.getValue()[1], (GLfloat)color.getValue()[2], 1.0 };
    glLightfv(GL_LIGHT0+lightID, GL_AMBIENT, c);
    glLightfv(GL_LIGHT0+lightID, GL_DIFFUSE, c);
    glLightfv(GL_LIGHT0+lightID, GL_SPECULAR, c);
    glLightf(GL_LIGHT0+lightID, GL_LINEAR_ATTENUATION, 0.0);

}

void Light::preDrawShadow(core::visual::VisualParams* /* vp */)
{
    if (needUpdate)
        updateVisual();
    const Vector3& pos = getPosition();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

#ifdef SOFA_HAVE_GLEW
    depthShader->setFloat(0, "zFar", (GLfloat) p_zFar.getValue());
    depthShader->setFloat(0, "zNear", (GLfloat) p_zNear.getValue());
    depthShader->setFloat4(0, "lightPosition", (GLfloat) pos[0], (GLfloat)pos[1], (GLfloat)pos[2], 1.0);
    depthShader->start();
    shadowFBO.start();
#endif
}

void Light::postDrawShadow()
{
#ifdef SOFA_HAVE_GLEW
    //Unbind fbo
    shadowFBO.stop();
    depthShader->stop();
#endif

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    if(softShadows.getValue())
        blurDepthTexture();
}

void Light::blurDepthTexture()
{
#ifdef SOFA_HAVE_GLEW
    float vxmax, vymax;
    float vxmin, vymin;
    float txmax, tymax;
    float txmin, tymin;

    txmin = tymin = 0.0;
    vxmin = vymin = -1.0;
    vxmax = vymax = txmax = tymax = 1.0;

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    blurHFBO.start();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, shadowFBO.getColorTexture());

    blurShader->setFloat(0, "mapDimX", (GLfloat) shadowTexWidth);
    blurShader->setInt(0, "orientation", 0);
    blurShader->start();

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin,tymax,0.0); glVertex3f(vxmin,vymax,0.0);
        glTexCoord3f(txmax,tymax,0.0); glVertex3f(vxmax,vymax,0.0);
        glTexCoord3f(txmax,tymin,0.0); glVertex3f(vxmax,vymin,0.0);
        glTexCoord3f(txmin,tymin,0.0); glVertex3f(vxmin,vymin,0.0);
    }
    glEnd();
    blurShader->stop();
    glBindTexture(GL_TEXTURE_2D, 0);

    blurHFBO.stop();

    blurVFBO.start();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, blurHFBO.getColorTexture());

    blurShader->setFloat(0, "mapDimX", (GLfloat) shadowTexWidth);
    blurShader->setInt(0, "orientation", 1);
    blurShader->start();

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin,tymax,0.0); glVertex3f(vxmin,vymax,0.0);
        glTexCoord3f(txmax,tymax,0.0); glVertex3f(vxmax,vymax,0.0);
        glTexCoord3f(txmax,tymin,0.0); glVertex3f(vxmax,vymin,0.0);
        glTexCoord3f(txmin,tymin,0.0); glVertex3f(vxmin,vymin,0.0);
    }
    glEnd();
    blurShader->stop();
    glBindTexture(GL_TEXTURE_2D, 0);

    blurVFBO.stop();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
#endif
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

void DirectionalLight::draw(const core::visual::VisualParams* )
{

}

PositionalLight::PositionalLight()
    :fixed(initData(&fixed, (bool) false, "fixed", "Fix light position from the camera"))
    ,position(initData(&position, (Vector3) Vector3(-0.7,0.3,0.0), "position", "Set the position of the light"))
    ,attenuation(initData(&attenuation, (float) 0.0, "attenuation", "Set the attenuation of the light"))
{

}

PositionalLight::~PositionalLight()
{

}

void PositionalLight::drawLight()
{
    Light::drawLight();

    GLfloat pos[4];
    pos[0]=(GLfloat)(position.getValue()[0]);
    pos[1]=(GLfloat)(position.getValue()[1]);
    pos[2]=(GLfloat)(position.getValue()[2]);
    pos[3]=1.0; // positional
    if (fixed.getValue())
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glLightfv(GL_LIGHT0+lightID, GL_POSITION, pos);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }
    else
        glLightfv(GL_LIGHT0+lightID, GL_POSITION, pos);

    glLightf(GL_LIGHT0+lightID, GL_LINEAR_ATTENUATION, attenuation.getValue());

}

void PositionalLight::draw(const core::visual::VisualParams* vparams)
{
    if (drawSource.getValue() && vparams->displayFlags().getShowVisualModels())
    {
        Vector3 sceneMinBBox, sceneMaxBBox;
        sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)this->getContext(), sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        float scale = (float)((sceneMaxBBox - sceneMinBBox).norm());
        scale *= 0.01f;

        GLUquadric* quad = gluNewQuadric();
        const Vector3& pos = position.getValue();
        const Vector3& col = color.getValue();

        glDisable(GL_LIGHTING);
        glColor3fv((float*)col.ptr());

        glPushMatrix();
        glTranslated(pos[0], pos[1], pos[2]);
        gluSphere(quad, 1.0f*scale, 16, 16);
        glPopMatrix();

        glEnable(GL_LIGHTING);
    }
}



SpotLight::SpotLight():
    direction(initData(&direction, (Vector3) Vector3(0,0,-1), "direction", "Set the direction of the light")),
    cutoff(initData(&cutoff, (float) 30.0, "cutoff", "Set the angle (cutoff) of the spot")),
    exponent(initData(&exponent, (float) 20.0, "exponent", "Set the exponent of the spot")),
    lookat(initData(&lookat, false, "lookat", "If true, direction specify the point at which the spotlight should be pointed to"))
{

}

SpotLight::~SpotLight()
{

}

void SpotLight::drawLight()
{
    PositionalLight::drawLight();
    defaulttype::Vector3 d = direction.getValue();
    if (lookat.getValue()) d -= position.getValue();
    d.normalize();
    GLfloat dir[3]= {(GLfloat)(d[0]), (GLfloat)(d[1]), (GLfloat)(d[2])};
    if (fixed.getValue())
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
    }
    glLightf(GL_LIGHT0+lightID, GL_SPOT_CUTOFF, cutoff.getValue());
    glLightfv(GL_LIGHT0+lightID, GL_SPOT_DIRECTION, dir);
    glLightf(GL_LIGHT0+lightID, GL_SPOT_EXPONENT, exponent.getValue());
    if (fixed.getValue())
    {
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }
}

void SpotLight::draw(const core::visual::VisualParams* vparams)
{
    if (drawSource.getValue() && vparams->displayFlags().getShowVisualModels())
    {
        Vector3 sceneMinBBox, sceneMaxBBox;
        sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)this->getContext(), sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        float scale = (float)((sceneMaxBBox - sceneMinBBox).norm());
        scale *= 0.01f;
        float width = 5.0f;
        float base =(float)(tan(cutoff.getValue()*M_PI/360)*width*2);

        static GLUquadric* quad = gluNewQuadric();
        const Vector3& pos = position.getValue();
        Vector3 dir = direction.getValue();
        if (lookat.getValue()) dir -= position.getValue();

        const Vector3& col = color.getValue();

        //get Rotation
        Vector3 xAxis, yAxis;
        yAxis = (fabs(dir[1]) > fabs(dir[2])) ? Vector3(0.0,0.0,1.0) : Vector3(0.0,1.0,0.0);
        xAxis = yAxis.cross(dir);
        yAxis = dir.cross(xAxis);
        xAxis.normalize();
        yAxis.normalize();
        dir.normalize();
        GLfloat rotMat[16] =
        {
            (GLfloat)xAxis[0], (GLfloat)xAxis[1], (GLfloat)xAxis[2], 0.0f,
            (GLfloat)yAxis[0], (GLfloat)yAxis[1], (GLfloat)yAxis[2], 0.0f,
            (GLfloat)dir  [0], (GLfloat)dir  [1], (GLfloat)dir  [2], 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
        glDisable(GL_LIGHTING);
        glColor3fv((float*)col.ptr());

        glPushMatrix();
        glTranslated(pos[0], pos[1], pos[2]);
        glMultMatrixf(rotMat);
        glScalef(scale, scale, scale);
        glPushMatrix();
        gluSphere(quad, 0.5, 16, 16);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        gluCylinder(quad,0.0, base, width, 10, 10);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glPopMatrix();
        glPopMatrix();

        glEnable(GL_LIGHTING);
    }
}

void SpotLight::preDrawShadow(core::visual::VisualParams* vp)
{
    double zNear=1e10, zFar=-1e10;

    Light::preDrawShadow(vp);
    const sofa::defaulttype::BoundingBox& sceneBBox = vp->sceneBBox();
    const Vector3 &pos = position.getValue();
    Vector3 dir = direction.getValue();
    if (lookat.getValue()) dir -= position.getValue();

    Vector3 xAxis, yAxis;

    yAxis=Vector3(0.0,1.0,0.0);

    if( 1.0 - std::abs(dot(yAxis, dir.normalized()))  < 0.0001)
    {
        dir += Vector3(0.0000001,0.0,0.0) * dot(yAxis, dir.normalized());
        dir.normalize();

    }
    xAxis = yAxis.cross(dir);
    xAxis.normalize();
    yAxis = dir.cross(xAxis);
    yAxis.normalize();

    defaulttype::Quat q;
    q = q.createQuaterFromFrame(xAxis, yAxis, dir);

    if (!p_zNear.isSet() || !p_zFar.isSet())
    {
        //compute zNear, zFar from light point of view
        for (int corner=0; corner<8; ++corner)
        {
            Vector3 p(
                (corner&1)?sceneBBox.minBBox().x():sceneBBox.maxBBox().x(),
                (corner&2)?sceneBBox.minBBox().y():sceneBBox.maxBBox().y(),
                (corner&4)?sceneBBox.minBBox().z():sceneBBox.maxBBox().z());
            p = q.rotate(p - pos);
            double z = -p[2];
            if (z < zNear) zNear = z;
            if (z > zFar)  zFar = z;
        }
        if (this->f_printLog.getValue())
            sout << "zNear = " << zNear << "  zFar = " << zFar << sendl;

        if (zNear <= 0)
            zNear = 1;
        if (zFar >= 1000.0)
            zFar = 1000.0;
        if (zFar <= 0)
        {
            zNear = vp->zNear();
            zFar = vp->zFar();
        }

        if (zNear > 0 && zFar < 1000)
        {
            zNear *= 0.8; // add some margin
            zFar *= 1.2;
            if (zNear < zFar*0.01)
                zNear = zFar*0.01;
            if (zNear < 0.1) zNear = 0.1;
            if (zFar < 2.0) zFar = 2.0;
        }

        p_zNear.setValue(zNear);
        p_zFar.setValue(zFar);
    }
    else
    {
        zNear = p_zNear.getValue();
        zFar = p_zFar.getValue();
    }

    //Projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(2.0*cutoff.getValue(),1.0, zNear, zFar);

    //Modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(pos[0], pos[1], pos[2],dir[0]+pos[0], dir[1]+pos[1], dir[2]+pos[2], 0.0,1.0,0.0);

    //Save the two matrices
    glGetFloatv(GL_PROJECTION_MATRIX, lightMatProj);
    glGetFloatv(GL_MODELVIEW_MATRIX, lightMatModelview);

    //glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, shadowFBO);

    glViewport(0, 0, shadowTexWidth, shadowTexHeight);

    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
}

GLuint SpotLight::getDepthTexture()
{
    //return debugVisualShadowTexture;
    //return shadowTexture;
#ifdef SOFA_HAVE_GLEW
    return shadowFBO.getDepthTexture();
#else
    return 0;
#endif
}

GLuint SpotLight::getColorTexture()
{
    //return debugVisualShadowTexture;
    //return shadowTexture;
#ifdef SOFA_HAVE_GLEW
    if(softShadows.getValue())
        return blurVFBO.getColorTexture();
    else
        return shadowFBO.getColorTexture();
#else
    return 0;
#endif
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
