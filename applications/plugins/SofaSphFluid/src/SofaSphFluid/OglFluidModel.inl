#pragma once

#include <SofaSphFluid/OglFluidModel.h>

#include <sstream>
#include <sofa/core/visual/VisualParams.h>
#include <limits>

#include <SofaSphFluid/shaders/pointToSprite.cppglsl>
#include <SofaSphFluid/shaders/spriteToSpriteNormal.cppglsl>
#include <SofaSphFluid/shaders/spriteBlurDepth.cppglsl>
#include <SofaSphFluid/shaders/spriteBlurThickness.cppglsl>
#include <SofaSphFluid/shaders/spriteShade.cppglsl>
#include <SofaSphFluid/shaders/spriteFinalPass.cppglsl>

namespace sofa
{
namespace component
{
namespace visualmodel
{

using namespace sofa::type;
using namespace sofa::defaulttype;

const float SPRITE_SCALE_DIV = tanf(65.0f * ((float)M_PI_2 / 180.0f));

template<class DataTypes>
OglFluidModel<DataTypes>::OglFluidModel()
    : m_positions(initData(&m_positions, "position", "Vertices coordinates"))
    , d_debugFBO(initData(&d_debugFBO, unsigned(9), "debugFBO", "DEBUG FBO"))
    , d_spriteRadius(initData(&d_spriteRadius, 1.0f,"spriteRadius", "Radius of sprites"))
    , d_spriteThickness(initData(&d_spriteThickness, 0.01f,"spriteThickness", "Thickness of sprites"))
    , d_spriteBlurRadius(initData(&d_spriteBlurRadius,  float(10.f), "spriteBlurRadius", "Blur radius (in pixels)"))
    , d_spriteBlurScale(initData(&d_spriteBlurScale, 0.1f, "spriteBlurScale", "Blur scale"))
    , d_spriteBlurDepthFalloff(initData(&d_spriteBlurDepthFalloff, 1.0f,"spriteBlurDepthFalloff", "Blur Depth Falloff"))
    , d_spriteDiffuseColor(initData(&d_spriteDiffuseColor, sofa::type::RGBAColor::blue(),"spriteDiffuseColor", "Diffuse Color"))
{
}

template<class DataTypes>
OglFluidModel<DataTypes>::~OglFluidModel()
{
}

template<class DataTypes>
void OglFluidModel<DataTypes>::init()
{ 
    VisualModel::init();
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template<class DataTypes>
void OglFluidModel<DataTypes>::initVisual()
{
    m_spriteDepthFBO = new sofa::gl::FrameBufferObject(true, true, true);
    m_spriteThicknessFBO = new sofa::gl::FrameBufferObject(true, true, true);
    m_spriteNormalFBO = new sofa::gl::FrameBufferObject(true, true, true);
    m_spriteBlurDepthHFBO = new sofa::gl::FrameBufferObject(true, true, true);
    m_spriteBlurDepthVFBO = new sofa::gl::FrameBufferObject(true, true, true);
    m_spriteBlurThicknessHFBO = new sofa::gl::FrameBufferObject(true, true, true);
    m_spriteBlurThicknessVFBO = new sofa::gl::FrameBufferObject(true, true, true);
    m_spriteShadeFBO = new sofa::gl::FrameBufferObject(true, true, true);

    const VecCoord& vertices = m_positions.getValue();

    // should set a fixed size, or update FBO size if the window is resized
    sofa::core::visual::VisualParams* vparams = sofa::core::visual::visualparams::defaultInstance();
    const int width = vparams->viewport()[2];
    const int height = vparams->viewport()[3];

    m_spriteDepthFBO->init(width, height);
    m_spriteThicknessFBO->init(width, height);
    m_spriteNormalFBO->init(width, height);
    m_spriteBlurDepthHFBO->init(width, height);
    m_spriteBlurDepthVFBO->init(width, height);
    m_spriteBlurThicknessHFBO->init(width, height);
    m_spriteBlurThicknessVFBO->init(width, height);
    m_spriteShadeFBO->init(width, height);

    if (!sofa::gl::GLSLShader::InitGLSL())
    {
        msg_error() << "InitGLSL failed, check your GPU setup (driver, etc)";
        return;
    }
    m_spriteShader.SetVertexShaderFromString(shader::pointToSpriteVS);
    m_spriteShader.SetFragmentShaderFromString(shader::pointToSpriteFS);
    m_spriteShader.InitShaders();

    m_spriteNormalShader.SetVertexShaderFromString(shader::spriteToSpriteNormalVS);
    m_spriteNormalShader.SetFragmentShaderFromString(shader::spriteToSpriteNormalFS);
    m_spriteNormalShader.InitShaders();

    m_spriteBlurDepthShader.SetVertexShaderFromString(shader::spriteBlurDepthVS);
    m_spriteBlurDepthShader.SetFragmentShaderFromString(shader::spriteBlurDepthFS);
    m_spriteBlurDepthShader.InitShaders();

    m_spriteBlurThicknessShader.SetVertexShaderFromString(shader::spriteBlurThicknessVS);
    m_spriteBlurThicknessShader.SetFragmentShaderFromString(shader::spriteBlurThicknessFS);
    m_spriteBlurThicknessShader.InitShaders();

    m_spriteShadeShader.SetVertexShaderFromString(shader::spriteShadeVS);
    m_spriteShadeShader.SetFragmentShaderFromString(shader::spriteShadeFS);
    m_spriteShadeShader.InitShaders();

    m_spriteFinalPassShader.SetVertexShaderFromString(shader::spriteFinalPassVS);
    m_spriteFinalPassShader.SetFragmentShaderFromString(shader::spriteFinalPassFS);
    m_spriteFinalPassShader.InitShaders();

    //Generate PositionVBO
    glGenBuffers(1, &m_posVBO);
    size_t totalSize = (vertices.size() * sizeof(vertices[0]));
    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 totalSize,
                 NULL,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    updateVertexBuffer();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

template<class DataTypes>
void OglFluidModel<DataTypes>::updateVisual()
{
    m_positions.updateIfDirty();
    updateVertexBuffer();
}

template<class DataTypes>
void OglFluidModel<DataTypes>::fwdDraw(core::visual::VisualParams*)
{
    updateVisual();

}

template<class DataTypes>
void OglFluidModel<DataTypes>::bwdDraw(core::visual::VisualParams*)
{

}

template<class DataTypes>
void OglFluidModel<DataTypes>::drawSprites(const core::visual::VisualParams* vparams)
{
    const VecCoord& positions = m_positions.getValue();

    if (positions.size() < 1)
        return;

    // Fetch and prepare the matrices to be sent to the shaders
    double projMat[16];
    double modelMat[16];

    vparams->getProjectionMatrix(projMat);
    float fProjMat[16];
    for (unsigned int i = 0; i < 16; i++)
        fProjMat[i] = float(projMat[i]);

    vparams->getModelViewMatrix(modelMat);
    float fModelMat[16];
    for (unsigned int i = 0; i < 16; i++)
        fModelMat[i] = float(modelMat[i]);

    Mat4x4f matProj(fProjMat);
    Mat4x4f invmatProj;
    if (!invmatProj.invert(matProj))
    {
        msg_error() << "Rendering failed (drawSprites) as the current Projection Matrix is singular.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    const float zNear = float(vparams->zNear());
    const float zFar = float(vparams->zFar());

    const float width = float(vparams->viewport()[2]);
    const float height = float(vparams->viewport()[3]);

    const float clearColor[4] = { 1.0f,1.0f,1.0f, 1.0f };
    ///////////////////////////////////////////////
    /// Sprites - Thickness

    m_spriteThicknessFBO->start();
    glClearColor(0.0, clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    std::vector<unsigned int> indices;
    for (unsigned int i = 0; i<positions.size(); i++)
        indices.push_back(i);
    ////// Compute sphere and depth

    m_spriteShader.TurnOn();

    m_spriteShader.SetMatrix4(m_spriteShader.GetVariable("u_projectionMatrix"), 1, false, fProjMat);
    m_spriteShader.SetMatrix4(m_spriteShader.GetVariable("u_modelviewMatrix"), 1, false, fModelMat);
    m_spriteShader.SetFloat(m_spriteShader.GetVariable("u_zNear"), zNear);
    m_spriteShader.SetFloat(m_spriteShader.GetVariable("u_zFar"), zFar);
    m_spriteShader.SetFloat(m_spriteShader.GetVariable("u_spriteRadius"), d_spriteRadius.getValue());
    m_spriteShader.SetFloat(m_spriteShader.GetVariable("u_spriteThickness"), d_spriteThickness.getValue());
    m_spriteShader.SetFloat(m_spriteShader.GetVariable("u_spriteScale"), (width / SPRITE_SCALE_DIV));

    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    glVertexPointer(3, GL_DOUBLE, 0, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    //
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDisable(GL_DEPTH_TEST);

    glDrawElements(GL_POINTS, GLsizei(positions.size()), GL_UNSIGNED_INT, &indices[0]);

    glDisableClientState(GL_VERTEX_ARRAY);

    m_spriteShader.TurnOff();

    m_spriteThicknessFBO->stop();

    glDisable(GL_BLEND);
    ///////////////////////////////////////////////
    /// Sprites - Depth

    m_spriteDepthFBO->start();
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_spriteShader.TurnOn();

    m_spriteShader.SetMatrix4(m_spriteShader.GetVariable("u_projectionMatrix"), 1, false, fProjMat);
    m_spriteShader.SetMatrix4(m_spriteShader.GetVariable("u_modelviewMatrix"), 1, false, fModelMat);
    m_spriteShader.SetFloat(m_spriteShader.GetVariable("u_zNear"), zNear);
    m_spriteShader.SetFloat(m_spriteShader.GetVariable("u_zFar"), zFar);
    m_spriteShader.SetFloat(m_spriteShader.GetVariable("u_spriteRadius"), d_spriteRadius.getValue());
    m_spriteShader.SetFloat(m_spriteShader.GetVariable("u_spriteThickness"), d_spriteThickness.getValue());
    m_spriteShader.SetFloat(m_spriteShader.GetVariable("u_spriteScale"), (width / SPRITE_SCALE_DIV));

    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    glVertexPointer(3, GL_DOUBLE, 0, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_DEPTH_TEST);
    glDrawElements(GL_POINTS, GLsizei(positions.size()), GL_UNSIGNED_INT, &indices[0]);

    glDisableClientState(GL_VERTEX_ARRAY);

    m_spriteShader.TurnOff();

    m_spriteDepthFBO->stop();

    glEnable(GL_DEPTH_TEST);

    ///////////////////////////////////////////////
    ////// Blur Depth texture (Horizontal)
    m_spriteBlurDepthHFBO->start();
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    m_spriteBlurDepthShader.TurnOn();

    m_spriteBlurDepthShader.SetInt(m_spriteBlurDepthShader.GetVariable("u_depthTexture"), 0);
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_width"), width);
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_height"), height);
    m_spriteBlurDepthShader.SetFloat2(m_spriteBlurDepthShader.GetVariable("u_direction"), 1, 0);
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_spriteBlurRadius"), d_spriteBlurRadius.getValue());
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_spriteBlurScale"), d_spriteBlurScale.getValue());
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_spriteBlurDepthFalloff"), d_spriteBlurDepthFalloff.getValue());
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_zNear"), zNear);
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_zFar"), zFar);

    float vxmax, vymax;
    float vxmin, vymin;
    float txmax, tymax;
    float txmin, tymin;

    txmin = tymin = 0.0;
    vxmin = vymin = -1.0;
    vxmax = vymax = txmax = tymax = 1.0;

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_spriteDepthFBO->getDepthTexture());

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin, tymax, 0.0); glVertex3f(vxmin, vymax, 0.0);
        glTexCoord3f(txmax, tymax, 0.0); glVertex3f(vxmax, vymax, 0.0);
        glTexCoord3f(txmax, tymin, 0.0); glVertex3f(vxmax, vymin, 0.0);
        glTexCoord3f(txmin, tymin, 0.0); glVertex3f(vxmin, vymin, 0.0);
    }
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    m_spriteBlurDepthShader.TurnOff();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    m_spriteBlurDepthHFBO->stop();

    ///////////////////////////////////////////////
    ////// Blur Depth texture (Vertical)
    m_spriteBlurDepthVFBO->start();
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    m_spriteBlurDepthShader.TurnOn();

    m_spriteBlurDepthShader.SetInt(m_spriteBlurDepthShader.GetVariable("u_depthTexture"), 0);
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_width"), width);
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_height"), height);
    m_spriteBlurDepthShader.SetFloat2(m_spriteBlurDepthShader.GetVariable("u_direction"), 0, 1);
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_spriteBlurRadius"), d_spriteBlurRadius.getValue());
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_spriteBlurScale"), d_spriteBlurScale.getValue());
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_spriteBlurDepthFalloff"), d_spriteBlurDepthFalloff.getValue());
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_zNear"), zNear);
    m_spriteBlurDepthShader.SetFloat(m_spriteBlurDepthShader.GetVariable("u_zFar"), zFar);

    txmin = tymin = 0.0;
    vxmin = vymin = -1.0;
    vxmax = vymax = txmax = tymax = 1.0;

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_spriteBlurDepthHFBO->getDepthTexture());

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin, tymax, 0.0); glVertex3f(vxmin, vymax, 0.0);
        glTexCoord3f(txmax, tymax, 0.0); glVertex3f(vxmax, vymax, 0.0);
        glTexCoord3f(txmax, tymin, 0.0); glVertex3f(vxmax, vymin, 0.0);
        glTexCoord3f(txmin, tymin, 0.0); glVertex3f(vxmin, vymin, 0.0);
    }
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    m_spriteBlurDepthShader.TurnOff();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    m_spriteBlurDepthVFBO->stop();

    ///////////////////////////////////////////////
    ////// Compute Normals
    m_spriteNormalFBO->start();
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    m_spriteNormalShader.TurnOn();

    m_spriteNormalShader.SetMatrix4(m_spriteNormalShader.GetVariable("u_InvProjectionMatrix"), 1, false, invmatProj.ptr());
    m_spriteNormalShader.SetInt(m_spriteNormalShader.GetVariable("u_depthTexture"), 0);
    m_spriteNormalShader.SetFloat(m_spriteNormalShader.GetVariable("u_width"), width);
    m_spriteNormalShader.SetFloat(m_spriteNormalShader.GetVariable("u_height"), height);
    m_spriteNormalShader.SetFloat(m_spriteNormalShader.GetVariable("u_zNear"), zNear);
    m_spriteNormalShader.SetFloat(m_spriteNormalShader.GetVariable("u_zFar"), zFar);

    txmin = tymin = 0.0;
    vxmin = vymin = -1.0;
    vxmax = vymax = txmax = tymax = 1.0;

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_spriteBlurDepthVFBO->getDepthTexture());

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin, tymax, 0.0); glVertex3f(vxmin, vymax, 0.0);
        glTexCoord3f(txmax, tymax, 0.0); glVertex3f(vxmax, vymax, 0.0);
        glTexCoord3f(txmax, tymin, 0.0); glVertex3f(vxmax, vymin, 0.0);
        glTexCoord3f(txmin, tymin, 0.0); glVertex3f(vxmin, vymin, 0.0);
    }
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    m_spriteNormalShader.TurnOff();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    m_spriteNormalFBO->stop();


    ///////////////////////////////////////////////
    ////// Blur Thickness texture (Horizontal)
    m_spriteBlurThicknessHFBO->start();
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    m_spriteBlurThicknessShader.TurnOn();

    m_spriteBlurThicknessShader.SetInt(m_spriteBlurThicknessShader.GetVariable("u_depthTexture"), 0);
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_width"), width);
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_height"), height);
    m_spriteBlurThicknessShader.SetFloat2(m_spriteBlurThicknessShader.GetVariable("u_direction"), 1, 0);
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_spriteBlurRadius"), d_spriteBlurRadius.getValue());
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_spriteBlurScale"), d_spriteBlurScale.getValue());
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_spriteBlurDepthFalloff"), d_spriteBlurDepthFalloff.getValue());
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_zNear"), zNear);
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_zFar"), zFar);

    txmin = tymin = 0.0;
    vxmin = vymin = -1.0;
    vxmax = vymax = txmax = tymax = 1.0;

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_spriteThicknessFBO->getColorTexture());

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin, tymax, 0.0); glVertex3f(vxmin, vymax, 0.0);
        glTexCoord3f(txmax, tymax, 0.0); glVertex3f(vxmax, vymax, 0.0);
        glTexCoord3f(txmax, tymin, 0.0); glVertex3f(vxmax, vymin, 0.0);
        glTexCoord3f(txmin, tymin, 0.0); glVertex3f(vxmin, vymin, 0.0);
    }
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    m_spriteBlurThicknessShader.TurnOff();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    m_spriteBlurThicknessHFBO->stop();
    ///////////////////////////////////////////////
    ////// Blur Thickness texture (Vertical)
    m_spriteBlurThicknessVFBO->start();
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    m_spriteBlurThicknessShader.TurnOn();

    m_spriteBlurThicknessShader.SetInt(m_spriteBlurThicknessShader.GetVariable("u_depthTexture"), 0);
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_width"), width);
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_height"), height);
    m_spriteBlurThicknessShader.SetFloat2(m_spriteBlurThicknessShader.GetVariable("u_direction"), 0, 1);
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_spriteBlurRadius"), d_spriteBlurRadius.getValue());
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_spriteBlurScale"), d_spriteBlurScale.getValue());
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_spriteBlurDepthFalloff"), d_spriteBlurDepthFalloff.getValue());
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_zNear"), zNear);
    m_spriteBlurThicknessShader.SetFloat(m_spriteBlurThicknessShader.GetVariable("u_zFar"), zFar);

    txmin = tymin = 0.0;
    vxmin = vymin = -1.0;
    vxmax = vymax = txmax = tymax = 1.0;

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_spriteBlurThicknessHFBO->getColorTexture());

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin, tymax, 0.0); glVertex3f(vxmin, vymax, 0.0);
        glTexCoord3f(txmax, tymax, 0.0); glVertex3f(vxmax, vymax, 0.0);
        glTexCoord3f(txmax, tymin, 0.0); glVertex3f(vxmax, vymin, 0.0);
        glTexCoord3f(txmin, tymin, 0.0); glVertex3f(vxmin, vymin, 0.0);
    }
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    m_spriteBlurThicknessShader.TurnOff();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    m_spriteBlurThicknessVFBO->stop();
    ///////////////////////////////////////////////
    ////// Shade sprites
    m_spriteShadeFBO->start();
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    const sofa::type::RGBAColor& diffuse = d_spriteDiffuseColor.getValue();

    m_spriteShadeShader.TurnOn();

    m_spriteShadeShader.SetMatrix4(m_spriteShadeShader.GetVariable("u_InvProjectionMatrix"), 1, true, invmatProj.ptr());
    m_spriteShadeShader.SetInt(m_spriteShadeShader.GetVariable("u_normalTexture"), 0);
    m_spriteShadeShader.SetInt(m_spriteShadeShader.GetVariable("u_depthTexture"), 1);
    m_spriteShadeShader.SetInt(m_spriteShadeShader.GetVariable("u_thicknessTexture"), 2);
    m_spriteShadeShader.SetFloat(m_spriteShadeShader.GetVariable("u_width"), width);
    m_spriteShadeShader.SetFloat(m_spriteShadeShader.GetVariable("u_height"), height);
    m_spriteShadeShader.SetFloat4(m_spriteShadeShader.GetVariable("u_diffuseColor"), diffuse[0], diffuse[1], diffuse[2], diffuse[3]);

    txmin = tymin = 0.0;
    vxmin = vymin = -1.0;
    vxmax = vymax = txmax = tymax = 1.0;

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_spriteNormalFBO->getColorTexture());
    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_spriteBlurDepthVFBO->getDepthTexture());
    glActiveTexture(GL_TEXTURE2);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_spriteBlurThicknessVFBO->getColorTexture());
    //glBindTexture(GL_TEXTURE_2D, m_spriteFBO->getColorTexture());

    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin, tymax, 0.0); glVertex3f(vxmin, vymax, 0.0);
        glTexCoord3f(txmax, tymax, 0.0); glVertex3f(vxmax, vymax, 0.0);
        glTexCoord3f(txmax, tymin, 0.0); glVertex3f(vxmax, vymin, 0.0);
        glTexCoord3f(txmin, tymin, 0.0); glVertex3f(vxmin, vymin, 0.0);
    }
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    m_spriteShadeShader.TurnOff();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    m_spriteShadeFBO->stop();
}

template<class DataTypes>
void OglFluidModel<DataTypes>::doDrawVisual(const core::visual::VisualParams* vparams)
{
    float vxmax, vymax;
    float vxmin, vymin;
    float txmax, tymax;
    float txmin, tymin;

    txmin = tymin = 0.0;
    vxmin = vymin = -1.0;
    vxmax = vymax = txmax = tymax = 1.0;

    drawSprites(vparams);

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    const auto debugFBO = d_debugFBO.getValue();

    if (debugFBO >= 0 && debugFBO < 9)
    {
        glActiveTexture(GL_TEXTURE0);
        glEnable(GL_TEXTURE_2D);
        switch (debugFBO)
        {
        case 0:
            glBindTexture(GL_TEXTURE_2D, m_spriteThicknessFBO->getColorTexture());
            break;
        case 1:
            glBindTexture(GL_TEXTURE_2D, m_spriteDepthFBO->getDepthTexture());
            break;
        case 2:
            glBindTexture(GL_TEXTURE_2D, m_spriteBlurDepthHFBO->getColorTexture());
            break;
        case 3:
            glBindTexture(GL_TEXTURE_2D, m_spriteBlurDepthVFBO->getColorTexture());
            break;
        case 4:
            glBindTexture(GL_TEXTURE_2D, m_spriteNormalFBO->getColorTexture());
            break;
        case 5:
            glBindTexture(GL_TEXTURE_2D, m_spriteBlurThicknessHFBO->getColorTexture());
            break;
        case 6:
            glBindTexture(GL_TEXTURE_2D, m_spriteBlurThicknessVFBO->getColorTexture());
            break;
        default:
            break;

        }

        glBegin(GL_QUADS);
        {
            glTexCoord3f(txmin, tymax, 0.0); glVertex3f(vxmin, vymax, 0.0);
            glTexCoord3f(txmax, tymax, 0.0); glVertex3f(vxmax, vymax, 0.0);
            glTexCoord3f(txmax, tymin, 0.0); glVertex3f(vxmax, vymin, 0.0);
            glTexCoord3f(txmin, tymin, 0.0); glVertex3f(vxmin, vymin, 0.0);
        }
        glEnd();

        glBindTexture(GL_TEXTURE_2D, 0);
    }
    
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}
template<class DataTypes>
void OglFluidModel<DataTypes>::drawTransparent(const core::visual::VisualParams* vparams)
{
    vparams->drawTool()->saveLastState();
    
    const auto debugFBO = d_debugFBO.getValue();
    if(debugFBO > 7)
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glEnable(GL_DEPTH_TEST);
        m_spriteFinalPassShader.TurnOn();
        glActiveTexture(GL_TEXTURE0);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, m_spriteShadeFBO->getColorTexture());
        glActiveTexture(GL_TEXTURE1);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, m_spriteShadeFBO->getDepthTexture());

        m_spriteFinalPassShader.SetInt(m_spriteFinalPassShader.GetVariable("u_colorTexture"), 0);
        m_spriteFinalPassShader.SetInt(m_spriteFinalPassShader.GetVariable("u_depthTexture"), 1);
        
        float vxmax, vymax;
        float vxmin, vymin;
        float txmax, tymax;
        float txmin, tymin;

        txmin = tymin = 0.0;
        vxmin = vymin = -1.0;
        vxmax = vymax = txmax = tymax = 1.0;
        glBegin(GL_QUADS);
        {
            glTexCoord3f(txmin, tymax, 0.0); glVertex3f(vxmin, vymax, 0.0);
            glTexCoord3f(txmax, tymax, 0.0); glVertex3f(vxmax, vymax, 0.0);
            glTexCoord3f(txmax, tymin, 0.0); glVertex3f(vxmax, vymin, 0.0);
            glTexCoord3f(txmin, tymin, 0.0); glVertex3f(vxmin, vymin, 0.0);
        }
        glEnd();

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        m_spriteFinalPassShader.TurnOff();

        glDisable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }

    vparams->drawTool()->restoreLastState();
}

template<class DataTypes>
void OglFluidModel<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);
    SOFA_UNUSED(onlyVisible);
    const VecCoord& position = m_positions.getValue();
    constexpr const SReal max_real { std::numeric_limits<SReal>::max() };
    constexpr const SReal min_real { std::numeric_limits<SReal>::lowest() };

    SReal maxBBox[3] = {min_real,min_real,min_real};
    SReal minBBox[3] = {max_real,max_real,max_real};

    for(unsigned int i=0 ; i<position.size() ; i++)
    {
        const Coord& v = position[i];
        for (unsigned j = 0; j < 3; j++)
        {
            if (minBBox[j] > v[j]) minBBox[j] = v[j];
            if (maxBBox[j] < v[j]) maxBBox[j] = v[j];
        }
    }

    this->f_bbox.setValue(sofa::type::TBoundingBox<SReal>(minBBox,maxBBox));

}

template<class DataTypes>
void OglFluidModel<DataTypes>::updateVertexBuffer()
{
    const VecCoord& vertices = m_positions.getValue();

    //Positions
    size_t totalSize = (vertices.size() * sizeof(vertices[0]));
    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 totalSize,
                 nullptr,
                 GL_DYNAMIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER,
                    0,
                    totalSize,
                    vertices.data());
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


}
}
}

