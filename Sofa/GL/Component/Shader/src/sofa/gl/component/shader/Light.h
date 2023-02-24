/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/gl/component/shader/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/gl/template.h>
#include <sofa/core/visual/VisualModel.h>

#include <sofa/gl/FrameBufferObject.h>
#include <sofa/gl/component/shader/OglShader.h>

namespace sofa::gl::component::shader
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
class SOFA_GL_COMPONENT_SHADER_API Light : public sofa::core::visual::VisualModel
{
public:
    enum LightType { DIRECTIONAL = 0, POSITIONAL = 1, SPOTLIGHT = 2 };

    SOFA_CLASS(Light, core::visual::VisualModel);
protected:
    GLint m_lightID;
    GLuint m_shadowTexWidth, m_shadowTexHeight;

    std::unique_ptr<sofa::gl::FrameBufferObject> m_shadowFBO;
    std::unique_ptr<sofa::gl::FrameBufferObject> m_blurHFBO;
    std::unique_ptr<sofa::gl::FrameBufferObject> m_blurVFBO;
    static const std::string PATH_TO_GENERATE_DEPTH_TEXTURE_VERTEX_SHADER;
    static const std::string PATH_TO_GENERATE_DEPTH_TEXTURE_FRAGMENT_SHADER;
    static const std::string PATH_TO_BLUR_TEXTURE_VERTEX_SHADER;
    static const std::string PATH_TO_BLUR_TEXTURE_FRAGMENT_SHADER;
    OglShader::SPtr m_depthShader;
    OglShader::SPtr m_blurShader;
    GLfloat m_lightMatProj[16];
    GLfloat m_lightMatModelview[16];

    void computeShadowMapSize();
    void blurDepthTexture();
public:
    Data<sofa::type::RGBAColor> d_color; ///< Set the color of the light. (default=[1.0,1.0,1.0,1.0])
    Data<GLuint> d_shadowTextureSize; ///< [Shadowing] Set size for shadow texture 
    Data<bool> d_drawSource; ///< Draw Light Source
    Data<float> d_zNear; ///< [Shadowing] Light's ZNear
    Data<float> d_zFar; ///< [Shadowing] Light's ZFar
    Data<bool> d_shadowsEnabled; ///< [Shadowing] Enable Shadow from this light
    Data<bool> d_softShadows; ///< [Shadowing] Turn on Soft Shadow from this light
    Data<float> d_shadowFactor; ///< [Shadowing] Shadow Factor (decrease/increase darkness)
    Data<float> d_VSMLightBleeding; ///< [Shadowing] (VSM only) Light bleeding paramter
    Data<float> d_VSMMinVariance; ///< [Shadowing] (VSM only) Minimum variance parameter
    Data<unsigned short> d_textureUnit; ///< [Shadowing] Texture unit for the genereated shadow texture

protected:
    Light();
    ~Light() override;
public:
    Data<type::vector<float> > d_modelViewMatrix; ///< [Shadowing] ModelView Matrix
    Data<type::vector<float> > d_projectionMatrix; ///< [Shadowing] Projection Matrix

    void setID(const GLint& id);

    //VisualModel
    void initVisual() override;
    void init() override;
    virtual void drawLight();
    void reinit() override;
    void updateVisual() override;

    /// Draw the light source from an external point of view.
    virtual void drawSource(const sofa::core::visual::VisualParams*) = 0;

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
    virtual const sofa::type::Vec3 getPosition() { return sofa::type::Vec3(0.0,0.0,0.0); }
    virtual unsigned short getShadowTextureUnit() { return d_textureUnit.getValue(); }
    virtual void setShadowTextureUnit(const unsigned short unit) { d_textureUnit.setValue(unit); }
    virtual type::Vec3 getDirection() { return type::Vec3(); }
    virtual float getShadowFactor() { return d_shadowFactor.getValue(); }
    virtual float getVSMLightBleeding() { return d_VSMLightBleeding.getValue(); }
    virtual float getVSMMinVariance() { return d_VSMMinVariance.getValue(); }
    virtual LightType getLightType() = 0;

protected:
    bool b_needUpdate;

};

class SOFA_GL_COMPONENT_SHADER_API DirectionalLight : public Light
{
public:
    SOFA_CLASS(DirectionalLight, Light);

    Data<sofa::type::Vec3> d_direction; ///< Set the direction of the light

    DirectionalLight();
    ~DirectionalLight() override;
    void preDrawShadow(core::visual::VisualParams* vp) override;
    void drawLight() override;
    void draw(const core::visual::VisualParams* vparams) override;
    void drawSource(const core::visual::VisualParams* vparams) override;
    GLuint getDepthTexture() override;
    GLuint getColorTexture() override;
    type::Vec3 getDirection() override { return d_direction.getValue(); }
    LightType getLightType() override { return LightType::DIRECTIONAL; }
private:
    void computeClippingPlane(const core::visual::VisualParams* vp, float& left, float& right, float& top, float& bottom, float& zNear, float& zFar);
    void computeOpenGLProjectionMatrix(GLfloat mat[16], float& left, float& right, float& top, float& bottom, float& zNear, float& zFar);
    void computeOpenGLModelViewMatrix(GLfloat lightMatModelview[16], const sofa::type::Vec3 &direction);

};

class SOFA_GL_COMPONENT_SHADER_API PositionalLight : public Light
{
public:
    SOFA_CLASS(PositionalLight, Light);

    Data<bool> d_fixed; ///< Fix light position from the camera
    Data<sofa::type::Vec3> d_position; ///< Set the position of the light
    Data<float> d_attenuation; ///< Set the attenuation of the light

    PositionalLight();
    ~PositionalLight() override;
    void drawLight() override;
    void draw(const core::visual::VisualParams* vparams) override;
    void drawSource(const core::visual::VisualParams*) override;
    const sofa::type::Vec3 getPosition() override { return d_position.getValue(); }
    LightType getLightType() override { return LightType::POSITIONAL; }
};

class SOFA_GL_COMPONENT_SHADER_API SpotLight : public PositionalLight
{
public:
    SOFA_CLASS(SpotLight, PositionalLight);

    Data<sofa::type::Vec3> d_direction; ///< Set the direction of the light
    Data<float> d_cutoff; ///< Set the angle (cutoff) of the spot
    Data<float> d_exponent; ///< Set the exponent of the spot
    Data<bool> d_lookat; ///< If true, direction specify the point at which the spotlight should be pointed to

    SpotLight();
    ~SpotLight() override;
    void drawLight() override;
    void draw(const core::visual::VisualParams* vparams) override;
    void drawSource(const core::visual::VisualParams* vparams) override;

    void preDrawShadow(core::visual::VisualParams*  vp) override;
    GLuint getDepthTexture() override;
    GLuint getColorTexture() override;
    type::Vec3 getDirection() override { return d_direction.getValue(); }
    LightType getLightType() override { return LightType::SPOTLIGHT; }

private:
    void computeClippingPlane(const core::visual::VisualParams* vp, float& zNear, float& zFar);
    void computeOpenGLProjectionMatrix(GLfloat mat[16], float width, float height, float fov, float zNear, float zFar);
    void computeOpenGLModelViewMatrix(GLfloat lightMatModelview[16], const sofa::type::Vec3 &position, const sofa::type::Vec3 &direction);

};

} // namespace sofa::gl::component::shader
