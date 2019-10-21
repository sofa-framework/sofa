#ifndef OglFluidModel_INL_
#define OglFluidModel_INL_

#include "OglFluidModel.h"

#include <sstream>
#include <sofa/helper/gl/GLSLShader.h>
#include <sofa/core/visual/VisualParams.h>
#include <limits>

namespace sofa
{
namespace component
{
namespace visualmodel
{

const float SPRITE_SCALE_DIV = tanf(65.0f * (0.5f * 3.1415926535f / 180.0f));

template<class DataTypes>
const std::string OglFluidModel<DataTypes>::PATH_TO_SPRITE_VERTEX_SHADER = "shaders/pointToSprite.vert";
template<class DataTypes>
const std::string OglFluidModel<DataTypes>::PATH_TO_SPRITE_FRAGMENT_SHADER = "shaders/pointToSprite.frag";
template<class DataTypes>
const std::string OglFluidModel<DataTypes>::PATH_TO_SPRITENORMAL_VERTEX_SHADER = "shaders/spriteToSpriteNormal.vert";
template<class DataTypes>
const std::string OglFluidModel<DataTypes>::PATH_TO_SPRITENORMAL_FRAGMENT_SHADER = "shaders/spriteToSpriteNormal.frag";
template<class DataTypes>
const std::string OglFluidModel<DataTypes>::PATH_TO_SPRITEBLURDEPTH_VERTEX_SHADER = "shaders/spriteBlurDepth.vert";
template<class DataTypes>
const std::string OglFluidModel<DataTypes>::PATH_TO_SPRITEBLURDEPTH_FRAGMENT_SHADER = "shaders/spriteBlurDepth.frag";
template<class DataTypes>
const std::string OglFluidModel<DataTypes>::PATH_TO_SPRITEBLURTHICKNESS_VERTEX_SHADER = "shaders/spriteBlurThickness.vert";
template<class DataTypes>
const std::string OglFluidModel<DataTypes>::PATH_TO_SPRITEBLURTHICKNESS_FRAGMENT_SHADER = "shaders/spriteBlurThickness.frag";
template<class DataTypes>
const std::string OglFluidModel<DataTypes>::PATH_TO_SPRITESHADE_VERTEX_SHADER = "shaders/spriteShade.vert";
template<class DataTypes>
const std::string OglFluidModel<DataTypes>::PATH_TO_SPRITESHADE_FRAGMENT_SHADER = "shaders/spriteShade.frag";

template<class DataTypes>
OglFluidModel<DataTypes>::OglFluidModel()
    : m_positions(initData(&m_positions, "position", "Vertices coordinates"))
    , d_debugFBO(initData(&d_debugFBO,  (unsigned int) 9,"debugFBO", "DEBUG FBO"))
    , d_spriteRadius(initData(&d_spriteRadius,  (float) 1.0,"spriteRadius", "Radius of sprites"))
    , d_spriteThickness(initData(&d_spriteThickness,  (float) 0.01,"spriteThickness", "Thickness of sprites"))
    , d_spriteBlurRadius(initData(&d_spriteBlurRadius,  (unsigned int) 10, "spriteBlurRadius", "Blur radius"))
    , d_spriteBlurScale(initData(&d_spriteBlurScale,  (float) 0.1, "spriteBlurScale", "Blur scale"))
    , d_spriteBlurDepthFalloff(initData(&d_spriteBlurDepthFalloff,  (float) 1000,"spriteBlurDepthFalloff", "Blur Depth Falloff"))
    , d_spriteDiffuseColor(initData(&d_spriteDiffuseColor, Vec4f(0.0,0.0,1.0,1.0),"spriteDiffuseColor", "Diffuse Color"))
    , m_spriteShader(sofa::core::objectmodel::New<OglShader>())
    , m_spriteNormalShader(sofa::core::objectmodel::New<OglShader>())
    , m_spriteBlurDepthShader(sofa::core::objectmodel::New<OglShader>())
	, m_spriteBlurThicknessShader(sofa::core::objectmodel::New<OglShader>())
    , m_spriteShadeShader(sofa::core::objectmodel::New<OglShader>())

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
}

template<class DataTypes>
void OglFluidModel<DataTypes>::initVisual()
{
    m_spriteDepthFBO = new helper::gl::FrameBufferObject(true, true, true);
    m_spriteThicknessFBO = new helper::gl::FrameBufferObject(true, true, true);
    m_spriteNormalFBO = new helper::gl::FrameBufferObject(true, true, true);
    m_spriteBlurDepthHFBO = new helper::gl::FrameBufferObject(true, true, true);
    m_spriteBlurDepthVFBO = new helper::gl::FrameBufferObject(true, true, true);
	m_spriteBlurThicknessHFBO = new helper::gl::FrameBufferObject(true, true, true);
	m_spriteBlurThicknessVFBO = new helper::gl::FrameBufferObject(true, true, true);
    m_spriteShadeFBO = new helper::gl::FrameBufferObject(true, true, true);

    const ResizableExtVector<Coord> tmpvertices = m_positions.getValue();
    ResizableExtVector<Vec3f> vertices;

    for(unsigned int i=0 ; i<tmpvertices.size() ; i++)
    {
        vertices.push_back(Vec3f(tmpvertices[i][0], tmpvertices[i][1], tmpvertices[i][2]));
    }

    sofa::core::visual::VisualParams* vparams = sofa::core::visual::VisualParams::defaultInstance();
    m_spriteDepthFBO->init(vparams->viewport()[2], vparams->viewport()[3]);
    m_spriteThicknessFBO->init(vparams->viewport()[2], vparams->viewport()[3]);
    m_spriteNormalFBO->init(vparams->viewport()[2], vparams->viewport()[3]);
    m_spriteBlurDepthHFBO->init(vparams->viewport()[2], vparams->viewport()[3]);
    m_spriteBlurDepthVFBO->init(vparams->viewport()[2], vparams->viewport()[3]);
	m_spriteBlurThicknessHFBO->init(vparams->viewport()[2], vparams->viewport()[3]);
	m_spriteBlurThicknessVFBO->init(vparams->viewport()[2], vparams->viewport()[3]);
    m_spriteShadeFBO->init(vparams->viewport()[2], vparams->viewport()[3]);
    //m_spriteFBO->init(512,512);

    m_spriteShader->vertFilename.addPath(PATH_TO_SPRITE_VERTEX_SHADER, true);
    m_spriteShader->fragFilename.addPath(PATH_TO_SPRITE_FRAGMENT_SHADER, true);
    m_spriteShader->init();
    m_spriteShader->initVisual();
    m_spriteNormalShader->vertFilename.addPath(PATH_TO_SPRITENORMAL_VERTEX_SHADER, true);
    m_spriteNormalShader->fragFilename.addPath(PATH_TO_SPRITENORMAL_FRAGMENT_SHADER, true);
    m_spriteNormalShader->init();
    m_spriteNormalShader->initVisual();
    m_spriteBlurDepthShader->vertFilename.addPath(PATH_TO_SPRITEBLURDEPTH_VERTEX_SHADER, true);
    m_spriteBlurDepthShader->fragFilename.addPath(PATH_TO_SPRITEBLURDEPTH_FRAGMENT_SHADER, true);
    m_spriteBlurDepthShader->init();
    m_spriteBlurDepthShader->initVisual();
	m_spriteBlurThicknessShader->vertFilename.addPath(PATH_TO_SPRITEBLURTHICKNESS_VERTEX_SHADER, true);
	m_spriteBlurThicknessShader->fragFilename.addPath(PATH_TO_SPRITEBLURTHICKNESS_FRAGMENT_SHADER, true);
	m_spriteBlurThicknessShader->init();
	m_spriteBlurThicknessShader->initVisual();
    m_spriteShadeShader->vertFilename.addPath(PATH_TO_SPRITESHADE_VERTEX_SHADER, true);
    m_spriteShadeShader->fragFilename.addPath(PATH_TO_SPRITESHADE_FRAGMENT_SHADER, true);
    m_spriteShadeShader->init();
    m_spriteShadeShader->initVisual();

    //Generate PositionVBO
    glGenBuffers(1, &m_posVBO);
    unsigned positionsBufferSize;
    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));
    unsigned int totalSize = positionsBufferSize;
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
    if (m_positions.getValue().size() < 1)
        return;

    const ResizableExtVector<Coord>& position = m_positions.getValue();

    float zNear = vparams->zNear();
    float zFar = vparams->zFar();

    float clearColor[4] = { 1.0f,1.0f,1.0f, 1.0f };

    ///////////////////////////////////////////////
    /// Sprites - Thickness
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    m_spriteThicknessFBO->start();
    glClearColor(0.0, clearColor[1], clearColor[2], clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    std::vector<unsigned int> indices;
    for (unsigned int i = 0; i<position.size(); i++)
        indices.push_back(i);
    ////// Compute sphere and depth
    double projMat[16];
    double modelMat[16];

    vparams->getProjectionMatrix(projMat);
    float fProjMat[16];
    for (unsigned int i = 0; i < 16; i++)
        fProjMat[i] = projMat[i];
    vparams->getModelViewMatrix(modelMat);
    float fModelMat[16];
    for (unsigned int i = 0; i < 16; i++)
        fModelMat[i] = modelMat[i];

    m_spriteShader->setMatrix4(0, "u_projectionMatrix", 1, false, fProjMat);
    //m_spriteShader->setMatrix4(0, "u_modelviewMatrix", 1, false, fModelMat);
    m_spriteShader->setFloat(0, "u_zNear", zNear);
    m_spriteShader->setFloat(0, "u_zFar",  zFar);
    m_spriteShader->setFloat(0, "u_spriteRadius",  d_spriteRadius.getValue());
    m_spriteShader->setFloat(0, "u_spriteThickness",  d_spriteThickness.getValue());
    m_spriteShader->setFloat(0, "u_spriteScale",  ( float(vparams->viewport()[2]) / SPRITE_SCALE_DIV));

    m_spriteShader->start();
    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    glVertexPointer(3, GL_FLOAT, 0, (char*)NULL + 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    //
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDisable(GL_DEPTH_TEST);

    glDrawElements(GL_POINTS, position.size(), GL_UNSIGNED_INT, &indices[0]);

    glDisableClientState(GL_VERTEX_ARRAY);

    m_spriteShader->stop();

    m_spriteThicknessFBO->stop();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glDisable(GL_BLEND);
    ///////////////////////////////////////////////
    /// Sprites - Depth
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    m_spriteDepthFBO->start();
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_spriteShader->setMatrix4(0, "u_projectionMatrix", 1, false, fProjMat);
    m_spriteShader->setFloat(0, "u_zNear", zNear);
    m_spriteShader->setFloat(0, "u_zFar",  zFar);
    m_spriteShader->setFloat(0, "u_spriteRadius",  d_spriteRadius.getValue());
	m_spriteShader->setFloat(0, "u_spriteThickness", d_spriteThickness.getValue());
    m_spriteShader->setFloat(0, "u_spriteScale",  float(vparams->viewport()[2]) / SPRITE_SCALE_DIV );

    m_spriteShader->start();
    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    glVertexPointer(3, GL_FLOAT, 0, (char*)NULL + 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_DEPTH_TEST);
    glDrawElements(GL_POINTS, position.size(), GL_UNSIGNED_INT, &indices[0]);

    glDisableClientState(GL_VERTEX_ARRAY);

    m_spriteShader->stop();

    m_spriteDepthFBO->stop();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

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

    m_spriteBlurDepthShader->setInt(0, "u_depthTexture", 0);
    m_spriteBlurDepthShader->setFloat(0, "u_width", vparams->viewport()[2]);
    m_spriteBlurDepthShader->setFloat(0, "u_height", vparams->viewport()[3]);
    m_spriteBlurDepthShader->setFloat2(0, "u_direction", 1, 0);
    m_spriteBlurDepthShader->setFloat(0, "u_spriteBlurRadius", d_spriteBlurRadius.getValue());
    m_spriteBlurDepthShader->setFloat(0, "u_spriteBlurScale", d_spriteBlurScale.getValue());
    m_spriteBlurDepthShader->setFloat(0, "u_spriteBlurDepthFalloff", d_spriteBlurDepthFalloff.getValue());
    m_spriteBlurDepthShader->setFloat(0, "u_zNear", zNear);
    m_spriteBlurDepthShader->setFloat(0, "u_zFar",  zFar);

    //std::cout <<  vparams->zNear() << " " <<  vparams->zFar() << std::endl;

    m_spriteBlurDepthShader->start();
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

    m_spriteBlurDepthShader->stop();
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

    m_spriteBlurDepthShader->setInt(0, "u_depthTexture", 0);
    m_spriteBlurDepthShader->setFloat(0, "u_width", vparams->viewport()[2]);
    m_spriteBlurDepthShader->setFloat(0, "u_height", vparams->viewport()[3]);
    m_spriteBlurDepthShader->setFloat2(0, "u_direction", 0, 1);
    m_spriteBlurDepthShader->setFloat(0, "u_spriteBlurRadius", d_spriteBlurRadius.getValue());
    m_spriteBlurDepthShader->setFloat(0, "u_spriteBlurScale", d_spriteBlurScale.getValue());
    m_spriteBlurDepthShader->setFloat(0, "u_spriteBlurDepthFalloff", d_spriteBlurDepthFalloff.getValue());
    m_spriteBlurDepthShader->setFloat(0, "u_zNear", zNear);
    m_spriteBlurDepthShader->setFloat(0, "u_zFar",  zFar);

    m_spriteBlurDepthShader->start();

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

    m_spriteBlurDepthShader->stop();
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

    Mat4x4f matProj(fProjMat);
    Mat4x4f invmatProj;
    invmatProj.invert(matProj);

    m_spriteNormalShader->setMatrix4(0, "u_InvProjectionMatrix", 1, false, invmatProj.ptr());
    m_spriteNormalShader->setInt(0, "u_depthTexture", 0);
    m_spriteNormalShader->setFloat(0, "u_width", vparams->viewport()[2]);
    m_spriteNormalShader->setFloat(0, "u_height", vparams->viewport()[3]);
    m_spriteNormalShader->setFloat(0, "u_zNear", zNear);
    m_spriteNormalShader->setFloat(0, "u_zFar",  zFar);

    m_spriteNormalShader->start();

    txmin = tymin = 0.0;
    vxmin = vymin = -1.0;
    vxmax = vymax = txmax = tymax = 1.0;

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_spriteBlurDepthVFBO->getDepthTexture());
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

    m_spriteNormalShader->stop();
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

	m_spriteBlurThicknessShader->setInt(0, "u_thicknessTexture", 0);
	m_spriteBlurThicknessShader->setFloat(0, "u_width", vparams->viewport()[2]);
	m_spriteBlurThicknessShader->setFloat(0, "u_height", vparams->viewport()[3]);
	m_spriteBlurThicknessShader->setFloat2(0, "u_direction", 1, 0);
	m_spriteBlurThicknessShader->setFloat(0, "u_spriteBlurRadius", d_spriteBlurRadius.getValue());
	m_spriteBlurThicknessShader->setFloat(0, "u_spriteBlurScale", d_spriteBlurScale.getValue());
	m_spriteBlurThicknessShader->setFloat(0, "u_spriteBlurDepthFalloff", d_spriteBlurDepthFalloff.getValue());
	m_spriteBlurThicknessShader->setFloat(0, "u_zNear", zNear);
	m_spriteBlurThicknessShader->setFloat(0, "u_zFar", zFar);

	m_spriteBlurThicknessShader->start();

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

	m_spriteBlurThicknessShader->stop();
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

	m_spriteBlurThicknessShader->setInt(0, "u_thicknessTexture", 0);
	m_spriteBlurThicknessShader->setFloat(0, "u_width", vparams->viewport()[2]);
	m_spriteBlurThicknessShader->setFloat(0, "u_height", vparams->viewport()[3]);
	m_spriteBlurThicknessShader->setFloat2(0, "u_direction", 0, 1);
	m_spriteBlurThicknessShader->setFloat(0, "u_spriteBlurRadius", d_spriteBlurRadius.getValue());
	m_spriteBlurThicknessShader->setFloat(0, "u_spriteBlurScale", d_spriteBlurScale.getValue());
	m_spriteBlurThicknessShader->setFloat(0, "u_spriteBlurDepthFalloff", d_spriteBlurDepthFalloff.getValue());
	m_spriteBlurThicknessShader->setFloat(0, "u_zNear", zNear);
	m_spriteBlurThicknessShader->setFloat(0, "u_zFar", zFar);

	m_spriteBlurThicknessShader->start();

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

	m_spriteBlurThicknessShader->stop();
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
    const Vec4f diffuse = d_spriteDiffuseColor.getValue();
    m_spriteShadeShader->setMatrix4(0, "u_InvProjectionMatrix", 1, true, invmatProj.ptr());
    m_spriteShadeShader->setInt(0, "u_normalTexture", 0);
    m_spriteShadeShader->setInt(0, "u_depthTexture", 1);
    m_spriteShadeShader->setInt(0, "u_thicknessTexture", 2);
    m_spriteShadeShader->setFloat(0, "u_width", vparams->viewport()[2]);
    m_spriteShadeShader->setFloat(0, "u_height", vparams->viewport()[3]);
    m_spriteShadeShader->setFloat4(0, "u_diffuseColor", diffuse[0], diffuse[1], diffuse[2], diffuse[3]);
    m_spriteShadeShader->start();

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

    m_spriteShadeShader->stop();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    m_spriteShadeFBO->stop();
}

template<class DataTypes>
void OglFluidModel<DataTypes>::drawVisual(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowVisualModels()) return;

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

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    switch(d_debugFBO.getValue())
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
		case 9:
		default:
			glBindTexture(GL_TEXTURE_2D, m_spriteShadeFBO->getColorTexture());
			break;

    }
    //glBindTexture(GL_TEXTURE_2D, m_spriteBlurDepthHFBO->getColorTexture());
    //glBindTexture(GL_TEXTURE_2D, m_spriteNormalFBO->getColorTexture());
    //glBindTexture(GL_TEXTURE_2D, m_spriteShadeFBO->getColorTexture());
    //glBindTexture(GL_TEXTURE_2D, m_spriteBlurDepthVFBO->getColorTexture());


    glBegin(GL_QUADS);
    {
        glTexCoord3f(txmin, tymax, 0.0); glVertex3f(vxmin, vymax, 0.0);
        glTexCoord3f(txmax, tymax, 0.0); glVertex3f(vxmax, vymax, 0.0);
        glTexCoord3f(txmax, tymin, 0.0); glVertex3f(vxmax, vymin, 0.0);
        glTexCoord3f(txmin, tymin, 0.0); glVertex3f(vxmin, vymin, 0.0);
    }
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

}

template<class DataTypes>
void OglFluidModel<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{

    const ResizableExtVector<Coord>& position = m_positions.getValue();
    const SReal max_real = std::numeric_limits<SReal>::max();
    const SReal min_real = std::numeric_limits<SReal>::lowest();

    SReal maxBBox[3] = {min_real,min_real,min_real};
    SReal minBBox[3] = {max_real,max_real,max_real};

    for(unsigned int i=0 ; i<position.size() ; i++)
    {
        const Coord& v = position[i];

        if (minBBox[0] > v[0]) minBBox[0] = v[0];
        if (minBBox[1] > v[1]) minBBox[1] = v[1];
        if (minBBox[2] > v[2]) minBBox[2] = v[2];
        if (maxBBox[0] < v[0]) maxBBox[0] = v[0];
        if (maxBBox[1] < v[1]) maxBBox[1] = v[1];
        if (maxBBox[2] < v[2]) maxBBox[2] = v[2];
    }

    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<SReal>(minBBox,maxBBox));

}

template<class DataTypes>
void OglFluidModel<DataTypes>::updateVertexBuffer()
{
    const ResizableExtVector<Coord>& tmpvertices = m_positions.getValue();
    ResizableExtVector<Vec3f> vertices;

    for(unsigned int i=0 ; i<tmpvertices.size() ; i++)
    {
        vertices.push_back(Vec3f(tmpvertices[i][0], tmpvertices[i][1], tmpvertices[i][2]));
    }

    //Positions
    unsigned positionsBufferSize;
    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));
    unsigned int totalSize = positionsBufferSize;
    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    glBufferData(GL_ARRAY_BUFFER,
            totalSize,
            nullptr,
            GL_DYNAMIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER,
            0,
            positionsBufferSize,
            vertices.data());
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


}
}
}

#endif //OglFluidModel_INL_
