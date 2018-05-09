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
#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(Light)

SOFA_DECL_CLASS(DirectionalLight)
//Register DirectionalLight in the Object Factory
int DirectionalLightClass = core::RegisterObject("A directional light illuminating the scene with parallel rays of light (can cast shadows).")
        .add< DirectionalLight >()
        ;

SOFA_DECL_CLASS(PositionalLight)
//Register PositionalLight in the Object Factory
int PositionalLightClass = core::RegisterObject("A positional light illuminating the scene."
                                                "The light has a location from which the ray are starting in all direction  (cannot cast shadows for now)")
        .add< PositionalLight >()
        ;

SOFA_DECL_CLASS(SpotLight)
//Register SpotLight in the Object Factory
int SpotLightClass = core::RegisterObject("A spot light illuminating the scene."
                                          "The light has a location and a illumination cone restricting the directions"
                                          "taken by the rays of light  (can cast shadows).")
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
    : m_lightID(0), m_shadowTexWidth(0),m_shadowTexHeight(0)
    , d_color(initData(&d_color, defaulttype::RGBAColor(1.0,1.0,1.0,1.0), "color", "Set the color of the light. (default=[1.0,1.0,1.0,1.0])"))
    , d_shadowTextureSize(initData(&d_shadowTextureSize, (GLuint)0, "shadowTextureSize", "[Shadowing] Set size for shadow texture "))
    , d_drawSource(initData(&d_drawSource, (bool) false, "drawSource", "Draw Light Source"))
    , d_zNear(initData(&d_zNear, "zNear", "[Shadowing] Light's ZNear"))
    , d_zFar(initData(&d_zFar, "zFar", "[Shadowing] Light's ZFar"))
    , d_shadowsEnabled(initData(&d_shadowsEnabled, (bool) true, "shadowsEnabled", "[Shadowing] Enable Shadow from this light"))
    , d_softShadows(initData(&d_softShadows, (bool) false, "softShadows", "[Shadowing] Turn on Soft Shadow from this light"))
    , d_shadowFactor(initData(&d_shadowFactor, (float) 1.0, "shadowFactor", "[Shadowing] Shadow Factor (decrease/increase darkness)"))
    , d_VSMLightBleeding(initData(&d_VSMLightBleeding, (float) 0.05, "VSMLightBleeding", "[Shadowing] (VSM only) Light bleeding paramter"))
    , d_VSMMinVariance(initData(&d_VSMMinVariance, (float) 0.001, "VSMMinVariance", "[Shadowing] (VSM only) Minimum variance parameter"))
    , d_textureUnit(initData(&d_textureUnit, (unsigned short) 1, "textureUnit", "[Shadowing] Texture unit for the genereated shadow texture"))
    , d_modelViewMatrix(initData(&d_modelViewMatrix, "modelViewMatrix", "[Shadowing] ModelView Matrix"))
    , d_projectionMatrix(initData(&d_projectionMatrix, "projectionMatrix", "[Shadowing] Projection Matrix"))
    , b_needUpdate(false)
{
    helper::vector<float>& wModelViewMatrix = *d_modelViewMatrix.beginEdit();
    helper::vector<float>& wProjectionMatrix = *d_projectionMatrix.beginEdit();

    wModelViewMatrix.resize(16);
    wProjectionMatrix.resize(16);

    d_modelViewMatrix.endEdit();
    d_projectionMatrix.endEdit();

    //Set Read-Only as we dont want to modify it with the GUI
    d_modelViewMatrix.setReadOnly(true);
    d_projectionMatrix.setReadOnly(true);
    d_shadowTextureSize.setReadOnly(true);
}

Light::~Light()
{
}

void Light::setID(const GLint& id)
{
    m_lightID = id;
}

void Light::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    LightManager* lm = context->core::objectmodel::BaseContext::get<LightManager>();

    if(lm)
    {
        msg_info() << "This light is now attached to the light manager: '"<< lm->getName() << "'.";
        lm->putLight(this);
        d_shadowsEnabled.setParent(&(lm->d_shadowsEnabled));
        d_softShadows.setParent(&(lm->d_softShadowsEnabled));
    }
    else
    {
        msg_warning() << "No LightManager found." ;
    }

    if(!d_shadowsEnabled.getValue() && d_softShadows.getValue()){
        if(d_softShadows.isSet() && d_shadowsEnabled.isSet()){
            msg_warning() << "Soft shadow is specified but 'shadowEnable' is set to false. " << msgendl
                                 "To remove this warning message you need to synchronize the softShadow & shadowEnable parameters." << msgendl ;
        }
    }

    if(!d_shadowsEnabled.getValue()){
        if( d_shadowTextureSize.isSet() ){
            msg_warning() << "Shadow is not enabled. The 'shadowTextureSize' parameter is not used but has been set." << msgendl
                                 "To remove this warning message you can:" << msgendl
                                 " - set the 'shadowEnabled' parameter to true." << msgendl
                                 " - unset the 'shadowTextureSize' values.";
        }

        if( d_shadowFactor.isSet() ){
            msg_warning() << "Shadow is not enabled. The 'shadowFactor' parameter is not used but has been set." << msgendl
                                 "To remove this warning message you can:" << msgendl
                                 " - set the 'shadowEnabled' parameter to true." << msgendl
                                 " - unset the 'shadowFactor' values.";
        }

        if( d_textureUnit.isSet() ){
            msg_warning() << "Shadow is not enabled. The 'textureUnit' parameter is not used but has been set." << msgendl
                                 "To remove this warning message you can:" << msgendl
                                 " - set the 'shadowEnabled' parameter to true." << msgendl
                                 " - unset the 'textureUnit' values.";
        }

        if( d_zNear.isSet() ){
            msg_warning() << "Shadow is not enabled. The 'zNear' parameter is not used but has been set." << msgendl
                                 "To remove this warning message you can:" << msgendl
                                 " - set the 'shadowEnabled' parameter to true." << msgendl
                                 " - unset the 'zNear' values.";
        }

        if( d_zFar.isSet() ){
            msg_warning() << "Shadow is not enabled. The 'zFar' parameter is not used but has been set." << msgendl
                                 "To remove this warning message you can:" << msgendl
                                 " - set the 'shadowEnabled' parameter to true." << msgendl
                                 " - unset the 'zFar' values.";
        }
    }

    if(!d_softShadows.getValue()){
        if( d_VSMLightBleeding.isSet() ){
            msg_warning() << "Soft shadow is not enabled. The 'VSMLightBleeding' parameter is not used but has been set." << msgendl
                                 "To remove this warning message you can:" << msgendl
                                 " - set the 'softShadows' parameter to true." << msgendl
                                 " - unset the 'VSMLightBleeding' values.";
        }
        if( d_VSMMinVariance.isSet() ){
            msg_warning() << "Soft shadow is not enabled. The 'VSMMinVariance' parameter is not used but has been set." << msgendl
                                 "To remove this warning message you can:" << msgendl
                                 " - set the 'softShadows' parameter to true." << msgendl
                                 " - unset the 'VMSMinVariance' values.";
        }
    }

    if (!d_zNear.isSet())
    {
        d_zNear.setReadOnly(true);
    }
    if (!d_zFar.isSet())
    {
        d_zFar.setReadOnly(true);
    }
}

void Light::initVisual()
{
    //init Shadow part
    computeShadowMapSize();
    //Shadow part
    //Shadow texture init
#ifdef SOFA_HAVE_GLEW
    m_shadowFBO = std::unique_ptr<helper::gl::FrameBufferObject>(
                new helper::gl::FrameBufferObject(true, true, true));
    m_blurHFBO = std::unique_ptr<helper::gl::FrameBufferObject>(
                new helper::gl::FrameBufferObject(false,false,true));
    m_blurVFBO = std::unique_ptr<helper::gl::FrameBufferObject>(
                new helper::gl::FrameBufferObject(false,false,true));
    m_depthShader = sofa::core::objectmodel::New<OglShader>();
    m_blurShader = sofa::core::objectmodel::New<OglShader>();

    m_shadowFBO->init(m_shadowTexWidth, m_shadowTexHeight);
    m_blurHFBO->init(m_shadowTexWidth, m_shadowTexHeight);
    m_blurVFBO->init(m_shadowTexWidth, m_shadowTexHeight);
    m_depthShader->vertFilename.addPath(PATH_TO_GENERATE_DEPTH_TEXTURE_VERTEX_SHADER,true);
    m_depthShader->fragFilename.addPath(PATH_TO_GENERATE_DEPTH_TEXTURE_FRAGMENT_SHADER,true);
    m_depthShader->init();
    m_depthShader->initVisual();
    m_blurShader->vertFilename.addPath(PATH_TO_BLUR_TEXTURE_VERTEX_SHADER,true);
    m_blurShader->fragFilename.addPath(PATH_TO_BLUR_TEXTURE_FRAGMENT_SHADER,true);
    m_blurShader->init();
    m_blurShader->initVisual();
#endif // SOFA_HAVE_GLEW
}

void Light::updateVisual()
{
    if (!b_needUpdate) return;
    computeShadowMapSize();
    b_needUpdate = false;
}

void Light::reinit()
{
    b_needUpdate = true;
}

void Light::drawLight()
{
    if (b_needUpdate)
        updateVisual();
    glLightf(GL_LIGHT0+m_lightID, GL_SPOT_CUTOFF, 180.0);
    GLfloat c[4] = { (GLfloat)d_color.getValue()[0], (GLfloat)d_color.getValue()[1], (GLfloat)d_color.getValue()[2], 1.0 };
    glLightfv(GL_LIGHT0+m_lightID, GL_AMBIENT, c);
    glLightfv(GL_LIGHT0+m_lightID, GL_DIFFUSE, c);
    glLightfv(GL_LIGHT0+m_lightID, GL_SPECULAR, c);
    glLightf(GL_LIGHT0+m_lightID, GL_LINEAR_ATTENUATION, 0.0);

}

void Light::preDrawShadow(core::visual::VisualParams* /* vp */)
{
    if (b_needUpdate)
        updateVisual();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

#ifdef SOFA_HAVE_GLEW
    m_depthShader->setFloat(0, "u_zFar", this->getZFar());
    m_depthShader->setFloat(0, "u_zNear", this->getZNear());
    m_depthShader->setInt(0, "u_lightType", this->getLightType());
    m_depthShader->setFloat(0, "u_shadowFactor", d_shadowFactor.getValue());
    m_depthShader->start();
    m_shadowFBO->start();
#endif
}


const GLfloat* Light::getOpenGLProjectionMatrix()
{
    return m_lightMatProj;
}

const GLfloat* Light::getOpenGLModelViewMatrix()
{
    return m_lightMatModelview;
}



void Light::postDrawShadow()
{
#ifdef SOFA_HAVE_GLEW
    //Unbind fbo
    m_shadowFBO->stop();
    m_depthShader->stop();
#endif

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    if(d_softShadows.getValue())
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

    m_blurHFBO->start();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_shadowFBO->getColorTexture());

    m_blurShader->setFloat(0, "mapDimX", (GLfloat) m_shadowTexWidth);
    m_blurShader->setInt(0, "orientation", 0);
    m_blurShader->start();

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin,tymax,0.0); glVertex3f(vxmin,vymax,0.0);
        glTexCoord3f(txmax,tymax,0.0); glVertex3f(vxmax,vymax,0.0);
        glTexCoord3f(txmax,tymin,0.0); glVertex3f(vxmax,vymin,0.0);
        glTexCoord3f(txmin,tymin,0.0); glVertex3f(vxmin,vymin,0.0);
    }
    glEnd();
    m_blurShader->stop();
    glBindTexture(GL_TEXTURE_2D, 0);

    m_blurHFBO->stop();

    m_blurVFBO->start();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_blurHFBO->getColorTexture());

    m_blurShader->setFloat(0, "mapDimX", (GLfloat) m_shadowTexWidth);
    m_blurShader->setInt(0, "orientation", 1);
    m_blurShader->start();

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin,tymax,0.0); glVertex3f(vxmin,vymax,0.0);
        glTexCoord3f(txmax,tymax,0.0); glVertex3f(vxmax,vymax,0.0);
        glTexCoord3f(txmax,tymin,0.0); glVertex3f(vxmax,vymin,0.0);
        glTexCoord3f(txmin,tymin,0.0); glVertex3f(vxmin,vymin,0.0);
    }
    glEnd();
    m_blurShader->stop();
    glBindTexture(GL_TEXTURE_2D, 0);

    m_blurVFBO->stop();

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

    if (d_shadowTextureSize.getValue() <= 0)
    {
        //Get the size of the shadow map
        if (windowWidth >= 1024 && windowHeight >= 1024)
        {
            m_shadowTexWidth = m_shadowTexHeight = 1024;
        }
        else if (windowWidth >= 512 && windowHeight >= 512)
        {
            m_shadowTexWidth = m_shadowTexHeight = 512;
        }
        else if (windowWidth >= 256 && windowHeight >= 256)
        {
            m_shadowTexWidth = m_shadowTexHeight = 256;
        }
        else
        {
            m_shadowTexWidth = m_shadowTexHeight = 128;
        }
    }
    else
        m_shadowTexWidth = m_shadowTexHeight = d_shadowTextureSize.getValue();
}


GLuint Light::getShadowMapSize()
{
    return m_shadowTexWidth;
}

GLfloat Light::getZNear()
{
    return d_zNear.getValue();
}

GLfloat Light::getZFar()
{
    return d_zFar.getValue();
}

DirectionalLight::DirectionalLight()
    : d_direction(initData(&d_direction, (Vector3) Vector3(0,0,-1), "direction", "Set the direction of the light"))
{

}

DirectionalLight::~DirectionalLight()
{

}

void DirectionalLight::drawLight()
{
    Light::drawLight();
    GLfloat dir[4];

    dir[0]=(GLfloat)(d_direction.getValue()[0]);
    dir[1]=(GLfloat)(d_direction.getValue()[1]);
    dir[2]=(GLfloat)(d_direction.getValue()[2]);
    dir[3]=0.0; // directional

    glLightfv(GL_LIGHT0+m_lightID, GL_POSITION, dir);
}

void DirectionalLight::draw(const core::visual::VisualParams* )
{

}
void DirectionalLight::computeOpenGLModelViewMatrix(GLfloat mat[16], const sofa::defaulttype::Vector3 &direction)
{
    //1-compute bounding box
    sofa::core::visual::VisualParams* vp = sofa::core::visual::VisualParams::defaultInstance();
    const sofa::defaulttype::BoundingBox& sceneBBox = vp->sceneBBox();
    Vector3 center = (sceneBBox.minBBox() + sceneBBox.maxBBox()) * 0.5;
    Vector3 posLight = center;

    posLight[0] = sceneBBox.maxBBox()[0] - sceneBBox.minBBox()[0] * 0.5;
    posLight[1] = sceneBBox.maxBBox()[1] - sceneBBox.minBBox()[1] * 0.5;
    posLight[2] = sceneBBox.minBBox()[2];


    //2-compute orientation to fit the bbox from light's pov
    // bounding box in light space = frustrum
    double epsilon = 0.0000001;
    Vector3 zAxis = -direction;
    zAxis.normalize();
    Vector3 yAxis(0, 1, 0);

    if (fabs(zAxis[0]) < epsilon && fabs(zAxis[2]) < epsilon)
    {
        if (zAxis[1] > 0)
            yAxis = Vector3(0, 0, -1);
        else
            yAxis = Vector3(0, 0, 1);
    }

    Vector3 xAxis = yAxis.cross(zAxis);
    xAxis.normalize();

    yAxis = zAxis.cross(xAxis);
    yAxis.normalize();

    defaulttype::Quat q;
    q = q.createQuaterFromFrame(xAxis, yAxis, zAxis);
    for (unsigned int i = 0; i < 3; i++)
    {
        mat[i * 4] = xAxis[i];
        mat[i * 4 + 1] = yAxis[i];
        mat[i * 4 + 2] = zAxis[i];
    }

    //translation
    mat[12] = 0;
    mat[13] = 0;
    mat[14] = (sceneBBox.maxBBox()[2] - sceneBBox.minBBox()[2])*-0.5;

    //w
    mat[15] = 1;

    //Save output as data for external shaders
    //we transpose it to get a standard matrix (and not OpenGL formatted)
    helper::vector<float>& wModelViewMatrix = *d_modelViewMatrix.beginEdit();

    for (unsigned int i = 0; i < 4; i++)
        for (unsigned int j = 0; j < 4; j++)
        {
            wModelViewMatrix[i * 4 + j] = mat[j * 4 + i];
        }

    d_modelViewMatrix.endEdit();
}

void DirectionalLight::computeOpenGLProjectionMatrix(GLfloat mat[16], float& left, float& right, float& top, float& bottom, float& zNear, float& zFar)
{
    mat[0] = 2 / (right - left);
    mat[4] = 0.0;
    mat[8] = 0.0;
    mat[12] = -1 * (right + left) / (right - left);

    mat[1] = 0.0;
    mat[5] = 2 / (top - bottom);
    mat[9] = 0.0;
    mat[13] = -1 * (top + bottom) / (top - bottom);

    mat[2] = 0;
    mat[6] = 0;
    mat[10] = -2 / (zFar - zNear);
    mat[14] = -1 * (zFar + zNear) / (zFar - zNear);

    mat[3] = 0.0;
    mat[7] = 0.0;
    mat[11] = 0.0;
    mat[15] = 1.0;

    //Save output as data for external shaders
    //we transpose it to get a standard matrix (and not OpenGL formatted)
    helper::vector<float>& wProjectionMatrix = *d_projectionMatrix.beginEdit();

    for (unsigned int i = 0; i < 4; i++)
        for (unsigned int j = 0; j < 4; j++)
        {
            wProjectionMatrix[i * 4 + j] = mat[j * 4 + i];
        }

    d_projectionMatrix.endEdit();
}


void DirectionalLight::computeClippingPlane(const core::visual::VisualParams* vp, float& left, float& right, float& top, float& bottom, float& zNear, float& zFar )
{
    const sofa::defaulttype::BoundingBox& sceneBBox = vp->sceneBBox();
    Vector3 minBBox = sceneBBox.minBBox();
    Vector3 maxBBox = sceneBBox.maxBBox();

    float maxLength = (maxBBox - minBBox).norm();

    left = maxLength * -0.5;
    right = maxLength * 0.5;
    top = maxLength * 0.5;
    bottom = maxLength * -0.5;
    zNear = 0.0 - maxLength*0.01;
    zFar = maxLength;

    //if (d_zNear.isSet())
    //    zNear = d_zNear.getValue();
    //else
    d_zNear.setValue(zNear);

    //if (d_zFar.isSet())
    //    zFar = d_zFar.getValue();
    //else
    d_zFar.setValue(zFar);
}


void DirectionalLight::preDrawShadow(core::visual::VisualParams* vp)
{

    float zNear = -1e10, left= -1e10, bottom = -1e10;
    float zFar = 1e10, right = 1e10, top = 1e10;

    Light::preDrawShadow(vp);
    const Vector3& dir = d_direction.getValue();

    computeClippingPlane(vp, left, right, top, bottom, zNear, zFar);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    computeOpenGLProjectionMatrix(m_lightMatProj, left, right, top, bottom, zNear, zFar);
    glMultMatrixf(m_lightMatProj);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    computeOpenGLModelViewMatrix(m_lightMatModelview, dir);
    glMultMatrixf(m_lightMatModelview);

    glViewport(0, 0, m_shadowTexWidth, m_shadowTexHeight);

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
}

GLuint DirectionalLight::getDepthTexture()
{
    //return debugVisualShadowTexture;
    //return shadowTexture;
#ifdef SOFA_HAVE_GLEW
    return m_shadowFBO->getDepthTexture();
#else
    return 0;
#endif
}

GLuint DirectionalLight::getColorTexture()
{
#ifdef SOFA_HAVE_GLEW
    if (d_softShadows.getValue())
        return m_blurVFBO->getColorTexture();
    else
        return m_shadowFBO->getColorTexture();
#else
    return 0;
#endif
}

PositionalLight::PositionalLight()
    : d_fixed(initData(&d_fixed, (bool) false, "fixed", "Fix light position from the camera"))
    , d_position(initData(&d_position, (Vector3) Vector3(-0.7,0.3,0.0), "position", "Set the position of the light"))
    , d_attenuation(initData(&d_attenuation, (float) 0.0, "attenuation", "Set the attenuation of the light"))
{

}

PositionalLight::~PositionalLight()
{

}

void PositionalLight::drawLight()
{
    Light::drawLight();

    GLfloat pos[4];
    pos[0]=(GLfloat)(d_position.getValue()[0]);
    pos[1]=(GLfloat)(d_position.getValue()[1]);
    pos[2]=(GLfloat)(d_position.getValue()[2]);
    pos[3]=1.0; // positional
    if (d_fixed.getValue())
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glLightfv(GL_LIGHT0+m_lightID, GL_POSITION, pos);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }
    else
        glLightfv(GL_LIGHT0+m_lightID, GL_POSITION, pos);

    glLightf(GL_LIGHT0+m_lightID, GL_LINEAR_ATTENUATION, d_attenuation.getValue());

}

void PositionalLight::draw(const core::visual::VisualParams* vparams)
{
    if (d_drawSource.getValue() && vparams->displayFlags().getShowVisualModels())
    {
        Vector3 sceneMinBBox, sceneMaxBBox;
        sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)this->getContext(), sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        float scale = (float)((sceneMaxBBox - sceneMinBBox).norm());
        scale *= 0.01f;

        GLUquadric* quad = gluNewQuadric();
        const Vector3& pos = d_position.getValue();
        const auto& col = d_color.getValue();

        glDisable(GL_LIGHTING);
        glColor3fv((float*)col.data());

        glPushMatrix();
        glTranslated(pos[0], pos[1], pos[2]);
        gluSphere(quad, 1.0f*scale, 16, 16);
        glPopMatrix();

        glEnable(GL_LIGHTING);
    }
}



SpotLight::SpotLight()
    : d_direction(initData(&d_direction, (Vector3) Vector3(0,0,-1), "direction", "Set the direction of the light"))
    , d_cutoff(initData(&d_cutoff, (float) 30.0, "cutoff", "Set the angle (cutoff) of the spot"))
    , d_exponent(initData(&d_exponent, (float) 1.0, "exponent", "Set the exponent of the spot"))
    , d_lookat(initData(&d_lookat, false, "lookat", "If true, direction specify the point at which the spotlight should be pointed to"))
{

}

SpotLight::~SpotLight()
{

}

void SpotLight::drawLight()
{
    PositionalLight::drawLight();
    defaulttype::Vector3 d = d_direction.getValue();
    if (d_lookat.getValue()) d -= d_position.getValue();
    d.normalize();
    GLfloat dir[3]= {(GLfloat)(d[0]), (GLfloat)(d[1]), (GLfloat)(d[2])};
    if (d_fixed.getValue())
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
    }
    glLightf(GL_LIGHT0+m_lightID, GL_SPOT_CUTOFF, d_cutoff.getValue());
    glLightfv(GL_LIGHT0+m_lightID, GL_SPOT_DIRECTION, dir);
    glLightf(GL_LIGHT0+m_lightID, GL_SPOT_EXPONENT, d_exponent.getValue());
    if (d_fixed.getValue())
    {
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }

}

void SpotLight::draw(const core::visual::VisualParams* vparams)
{
    float zNear, zFar;

    computeClippingPlane(vparams, zNear, zFar);

    Vector3 dir = d_direction.getValue();
    if (d_lookat.getValue())
        dir -= d_position.getValue();

    computeOpenGLProjectionMatrix(m_lightMatProj, m_shadowTexWidth, m_shadowTexHeight, 2 * d_cutoff.getValue(), zNear, zFar);
    computeOpenGLModelViewMatrix(m_lightMatModelview, d_position.getValue(), dir);

    if (d_drawSource.getValue() && vparams->displayFlags().getShowVisualModels())
    {
        float baseLength = zFar * tanf(this->d_cutoff.getValue() * M_PI / 180);
        float tipLength = (baseLength*0.5) * (zNear/ zFar);

        Vector3 direction;
        if(d_lookat.getValue())
            direction = this->d_direction.getValue() - this->d_position.getValue();
        else
            direction = this->d_direction.getValue();

        direction.normalize();
        Vector3 base = this->getPosition() + direction*zFar;
        Vector3 tip = this->getPosition() + direction*zNear;
        std::vector<Vector3> centers;
        centers.push_back(this->getPosition());
        vparams->drawTool()->setPolygonMode(0, true);
        vparams->drawTool()->setLightingEnabled(false);
        vparams->drawTool()->drawSpheres(centers, zNear*0.1,d_color.getValue());
        vparams->drawTool()->drawCone(base, tip, baseLength, tipLength, d_color.getValue());
        vparams->drawTool()->setLightingEnabled(true);
        vparams->drawTool()->setPolygonMode(0, false);
    }
}

void SpotLight::computeClippingPlane(const core::visual::VisualParams* vp, float& zNear, float& zFar)
{
    zNear = 1e10;
    zFar = -1e10;

    const sofa::defaulttype::BoundingBox& sceneBBox = vp->sceneBBox();
    const Vector3 &pos = d_position.getValue();
    Vector3 dir = d_direction.getValue();
    if (d_lookat.getValue())
        dir -= d_position.getValue();

    double epsilon = 0.0000001;
    Vector3 zAxis = -dir;
    zAxis.normalize();
    Vector3 yAxis(0, 1, 0);

    if (fabs(zAxis[0]) < epsilon && fabs(zAxis[2]) < epsilon)
    {
        if (zAxis[1] > 0)
            yAxis = Vector3(0, 0, -1);
        else
            yAxis = Vector3(0, 0, 1);
    }

    Vector3 xAxis = yAxis.cross(zAxis);
    xAxis.normalize();

    yAxis = zAxis.cross(xAxis);
    yAxis.normalize();

    defaulttype::Quat q;
    q = q.createQuaterFromFrame(xAxis, yAxis, dir);

    if (!d_zNear.isSet() || !d_zFar.isSet())
    {
        //compute zNear, zFar from light point of view
        for (int corner = 0; corner<8; ++corner)
        {
            Vector3 p(
                        (corner & 1) ? sceneBBox.minBBox().x() : sceneBBox.maxBBox().x(),
                        (corner & 2) ? sceneBBox.minBBox().y() : sceneBBox.maxBBox().y(),
                        (corner & 4) ? sceneBBox.minBBox().z() : sceneBBox.maxBBox().z());
            p = q.rotate(p - pos);
            double z = -p[2];
            if (z < zNear) zNear = z;
            if (z > zFar)  zFar = z;
        }
        msg_info() << "zNear = " << zNear << "  zFar = " << zFar ;

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

        d_zNear.setValue(zNear);
        d_zFar.setValue(zFar);
    }
    else
    {
        zNear = d_zNear.getValue();
        zFar = d_zFar.getValue();
    }
}

void SpotLight::preDrawShadow(core::visual::VisualParams* vp)
{

    float zNear = -1e10, zFar = 1e10;

    const Vector3 &pos = d_position.getValue();
    Vector3 dir = d_direction.getValue();
    if (d_lookat.getValue())
        dir -= d_position.getValue();

    Light::preDrawShadow(vp);

    computeClippingPlane(vp, zNear, zFar);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    computeOpenGLProjectionMatrix(m_lightMatProj, m_shadowTexWidth, m_shadowTexHeight, 2 * d_cutoff.getValue(), zNear, zFar);
    glMultMatrixf(m_lightMatProj);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    computeOpenGLModelViewMatrix(m_lightMatModelview, pos, dir);
    glMultMatrixf(m_lightMatModelview);

    glViewport(0, 0, m_shadowTexWidth, m_shadowTexHeight);

    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
}

void SpotLight::computeOpenGLModelViewMatrix(GLfloat mat[16], const sofa::defaulttype::Vector3 &position, const sofa::defaulttype::Vector3 &direction)
{
    double epsilon = 0.0000001;
    Vector3 zAxis = -direction;
    zAxis.normalize();
    Vector3 yAxis(0, 1, 0);

    if (fabs(zAxis[0]) < epsilon && fabs(zAxis[2]) < epsilon)
    {
        if (zAxis[1] > 0)
            yAxis = Vector3(0, 0, -1);
        else
            yAxis = Vector3(0, 0, 1);
    }

    Vector3 xAxis = yAxis.cross(zAxis);
    xAxis.normalize();

    yAxis = zAxis.cross(xAxis);
    yAxis.normalize();

    for (unsigned int i = 0; i < 3; i++)
    {
        mat[i * 4] = xAxis[i];
        mat[i * 4 + 1] = yAxis[i];
        mat[i * 4 + 2] = zAxis[i];
    }

    sofa::defaulttype::Quat q;
    q = sofa::defaulttype::Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);

    Vector3 origin = q.inverseRotate(-position);

    //translation
    mat[12] = origin[0];
    mat[13] = origin[1];
    mat[14] = origin[2];

    //w
    mat[15] = 1;

    //Save output as data for external shaders
    //we transpose it to get a standard matrix (and not OpenGL formatted)
    helper::vector<float>& wModelViewMatrix = *d_modelViewMatrix.beginEdit();

    for (unsigned int i = 0; i < 4; i++)
        for (unsigned int j = 0; j < 4; j++)
        {
            wModelViewMatrix[i * 4 + j] = mat[j * 4 + i];
        }

    d_modelViewMatrix.endEdit();
}


void SpotLight::computeOpenGLProjectionMatrix(GLfloat mat[16], float width, float height, float fov, float zNear, float zFar)
{
    float scale = 1.0 / tan(fov * M_PI / 180 * 0.5);
    float aspect = width / height;

    float pm00 = scale / aspect;
    float pm11 = scale;

    mat[0] = pm00; // FocalX
    mat[4] = 0.0;
    mat[8] = 0.0;
    mat[12] = 0.0;

    mat[1] = 0.0;
    mat[5] = pm11; // FocalY
    mat[9] = 0.0;
    mat[13] = 0.0;

    mat[2] = 0;
    mat[6] = 0;
    mat[10] = -(zFar + zNear) / (zFar - zNear);
    mat[14] = -2.0 * zFar * zNear / (zFar - zNear);;

    mat[3] = 0.0;
    mat[7] = 0.0;
    mat[11] = -1.0;
    mat[15] = 0.0;

    //Save output as data for external shaders
    //we transpose it to get a standard matrix (and not OpenGL formatted)
    helper::vector<float>& wProjectionMatrix = *d_projectionMatrix.beginEdit();

    for (unsigned int i = 0; i < 4; i++)
        for (unsigned int j = 0; j < 4; j++)
        {
            wProjectionMatrix[i * 4 + j] = mat[j * 4 + i];
        }

    d_projectionMatrix.endEdit();
}

GLuint SpotLight::getDepthTexture()
{
#ifdef SOFA_HAVE_GLEW
    return m_shadowFBO->getDepthTexture();
#else
    return 0;
#endif
}

GLuint SpotLight::getColorTexture()
{
#ifdef SOFA_HAVE_GLEW
    if(d_softShadows.getValue())
        return m_blurVFBO->getColorTexture();
    else
        return m_shadowFBO->getColorTexture();
#else
    return 0;
#endif
}

}

} //namespace component

} //namespace sofa
