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
#ifndef SOFA_COMPONENT_LIGHT
#define SOFA_COMPONENT_LIGHT
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RGBAColor.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/visual/VisualModel.h>

#ifdef SOFA_HAVE_GLEW
#include <sofa/helper/gl/FrameBufferObject.h>
#include <SofaOpenglVisual/OglShader.h>
#endif

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief Utility to cast Light into a Opengl scene.
 *
 *  This class must be used in a scene with one LightManager object.
 *  This abstract class defines lights (i.e basically id and color)
 *  The inherited lights are:
 *   - Directional light (direction);
 *   - Positional light (position);
 *   - Spot light (position, direction, cutoff...).
 *
 */
class SOFA_OPENGL_VISUAL_API Light : public sofa::core::visual::VisualModel
{
public:
    enum LightType { DIRECTIONAL = 0, POSITIONAL = 1, SPOTLIGHT = 2 };

    SOFA_CLASS(Light, core::visual::VisualModel);
protected:
    GLint m_lightID;
    GLuint m_shadowTexWidth, m_shadowTexHeight;

#ifdef SOFA_HAVE_GLEW
    std::unique_ptr<helper::gl::FrameBufferObject> m_shadowFBO;
    std::unique_ptr<helper::gl::FrameBufferObject> m_blurHFBO;
    std::unique_ptr<helper::gl::FrameBufferObject> m_blurVFBO;
    static const std::string PATH_TO_GENERATE_DEPTH_TEXTURE_VERTEX_SHADER;
    static const std::string PATH_TO_GENERATE_DEPTH_TEXTURE_FRAGMENT_SHADER;
    static const std::string PATH_TO_BLUR_TEXTURE_VERTEX_SHADER;
    static const std::string PATH_TO_BLUR_TEXTURE_FRAGMENT_SHADER;
    OglShader::SPtr m_depthShader;
    OglShader::SPtr m_blurShader;
#endif
    GLfloat m_lightMatProj[16];
    GLfloat m_lightMatModelview[16];

    void computeShadowMapSize();
    void blurDepthTexture();
public:
    Data<defaulttype::RGBAColor> d_color; ///< Set the color of the light. (default=[1.0,1.0,1.0,1.0])
    Data<GLuint> d_shadowTextureSize; ///< [Shadowing] Set size for shadow texture 
    Data<bool> d_drawSource; ///< Draw Light Source
    Data<double> d_zNear; ///< [Shadowing] Light's ZNear
    Data<double> d_zFar; ///< [Shadowing] Light's ZFar
    Data<bool> d_shadowsEnabled; ///< [Shadowing] Enable Shadow from this light
    Data<bool> d_softShadows; ///< [Shadowing] Turn on Soft Shadow from this light
    Data<float> d_shadowFactor; ///< [Shadowing] Shadow Factor (decrease/increase darkness)
    Data<float> d_VSMLightBleeding; ///< [Shadowing] (VSM only) Light bleeding paramter
    Data<float> d_VSMMinVariance; ///< [Shadowing] (VSM only) Minimum variance parameter
    Data<unsigned short> d_textureUnit; ///< [Shadowing] Texture unit for the genereated shadow texture

protected:
    Light();
    virtual ~Light();
public:
    Data<helper::vector<float> > d_modelViewMatrix; ///< [Shadowing] ModelView Matrix
    Data<helper::vector<float> > d_projectionMatrix; ///< [Shadowing] Projection Matrix

    void setID(const GLint& id);

    //VisualModel
    virtual void initVisual() override;
    void init() override;
    virtual void drawLight();
    virtual void reinit() override;
    virtual void updateVisual() override;

    GLfloat getZNear();
    GLfloat getZFar();

    //CastShadowModel
    virtual void preDrawShadow(core::visual::VisualParams* vp);
    virtual void postDrawShadow();
    virtual GLuint getShadowMapSize();
    const GLfloat* getOpenGLProjectionMatrix();
    const GLfloat* getOpenGLModelViewMatrix();
    virtual GLuint getDepthTexture() { return 0 ;}
    virtual GLuint getColorTexture() { return 0 ;}
    virtual const sofa::defaulttype::Vector3 getPosition() { return sofa::defaulttype::Vector3(0.0,0.0,0.0); }
    virtual unsigned short getShadowTextureUnit() { return d_textureUnit.getValue(); }
    virtual void setShadowTextureUnit(const unsigned short unit) { d_textureUnit.setValue(unit); }
    virtual defaulttype::Vector3 getDirection() { return defaulttype::Vector3(); }
    virtual float getShadowFactor() { return d_shadowFactor.getValue(); }
    virtual float getVSMLightBleeding() { return d_VSMLightBleeding.getValue(); }
    virtual float getVSMMinVariance() { return d_VSMMinVariance.getValue(); }
    virtual LightType getLightType() = 0;

protected:
    bool b_needUpdate;

};

class SOFA_OPENGL_VISUAL_API DirectionalLight : public Light
{
public:
    SOFA_CLASS(DirectionalLight, Light);

    Data<sofa::defaulttype::Vector3> d_direction; ///< Set the direction of the light

    DirectionalLight();
    virtual ~DirectionalLight();
    virtual void preDrawShadow(core::visual::VisualParams* vp) override;
    virtual void drawLight() override;
    virtual void draw(const core::visual::VisualParams* vparams) override;
    virtual GLuint getDepthTexture() override;
    virtual GLuint getColorTexture() override;
    virtual defaulttype::Vector3 getDirection() override { return d_direction.getValue(); }
    LightType getLightType() override { return LightType::DIRECTIONAL; }
private:
    void computeClippingPlane(const core::visual::VisualParams* vp, float& left, float& right, float& top, float& bottom, float& zNear, float& zFar);
    void computeOpenGLProjectionMatrix(GLfloat mat[16], float& left, float& right, float& top, float& bottom, float& zNear, float& zFar);
    void computeOpenGLModelViewMatrix(GLfloat lightMatModelview[16], const sofa::defaulttype::Vector3 &direction);

};

class SOFA_OPENGL_VISUAL_API PositionalLight : public Light
{
public:
    SOFA_CLASS(PositionalLight, Light);

    Data<bool> d_fixed; ///< Fix light position from the camera
    Data<sofa::defaulttype::Vector3> d_position; ///< Set the position of the light
    Data<float> d_attenuation; ///< Set the attenuation of the light

    PositionalLight();
    virtual ~PositionalLight();
    virtual void drawLight() override;
    virtual void draw(const core::visual::VisualParams* vparams) override;
    virtual const sofa::defaulttype::Vector3 getPosition() override { return d_position.getValue(); }
    LightType getLightType() override { return LightType::POSITIONAL; }
};

class SOFA_OPENGL_VISUAL_API SpotLight : public PositionalLight
{
public:
    SOFA_CLASS(SpotLight, PositionalLight);

    Data<sofa::defaulttype::Vector3> d_direction; ///< Set the direction of the light
    Data<float> d_cutoff; ///< Set the angle (cutoff) of the spot
    Data<float> d_exponent; ///< Set the exponent of the spot
    Data<bool> d_lookat; ///< If true, direction specify the point at which the spotlight should be pointed to

    SpotLight();
    virtual ~SpotLight();
    virtual void drawLight() override;
    virtual void draw(const core::visual::VisualParams* vparams) override;

    void preDrawShadow(core::visual::VisualParams*  vp) override;
    GLuint getDepthTexture() override;
    GLuint getColorTexture() override;
    defaulttype::Vector3 getDirection() override { return d_direction.getValue(); }
    LightType getLightType() override { return LightType::SPOTLIGHT; }

private:
    void computeClippingPlane(const core::visual::VisualParams* vp, float& zNear, float& zFar);
    void computeOpenGLProjectionMatrix(GLfloat mat[16], float width, float height, float fov, float zNear, float zFar);
    void computeOpenGLModelViewMatrix(GLfloat lightMatModelview[16], const sofa::defaulttype::Vector3 &position, const sofa::defaulttype::Vector3 &direction);

};

} //namespace visualmodel

} //namespace component

} //namespace sofa

#endif //SOFA_COMPONENT_LIGHT
