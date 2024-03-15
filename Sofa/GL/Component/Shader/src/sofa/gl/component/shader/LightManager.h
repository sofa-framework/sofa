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

#include <sofa/gl/component/shader/Light.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/type/Mat.h>
#include <sofa/type/RGBAColor.h>

#include <sofa/gl/component/shader/OglShadowShader.h>

namespace sofa::gl::component::shader
{

/**
 *  \brief Utility to manage lights into an Opengl scene
 *
 *  This class must be used with the Light class.
 *  It centralizes all the Lights and managed them.
 *
 */
class SOFA_GL_COMPONENT_SHADER_API LightManager : public core::visual::VisualManager
{
public:
    SOFA_CLASS(LightManager, core::visual::VisualManager);

private:
    std::vector<Light::SPtr>          m_lights;
    std::vector<type::Mat4x4f> m_lightModelViewMatrix;
    std::vector<type::Mat4x4f> m_lightProjectionMatrix;
    std::vector<unsigned short>       m_mapShadowTextureUnit;

    //OglShadowShader* shadowShader;
    type::vector<OglShadowShader::SPtr> m_shadowShaders;
    void makeShadowMatrix(unsigned int i);

public:
    enum { MAX_NUMBER_OF_LIGHTS = /*GL_MAX_LIGHTS*/ 8 };

    //TODO(dmarchal): sofa guidelines.
    Data<bool>                  d_shadowsEnabled; ///< Enable Shadow in the scene. (default=0)
    Data<bool>                  d_softShadowsEnabled; ///< If Shadows enabled, Enable Variance Soft Shadow in the scene. (default=0)
    Data<sofa::type::RGBAColor>  d_ambient; ///< Ambient lights contribution (Vec4f)(default=[0.0f,0.0f,0.0f,0.0f])
    Data<bool>                  d_drawIsEnabled; ///< enable/disable drawing of lights shadow textures. (default=false)

protected:
    LightManager();
    ~LightManager() override;

public:
    void init() override;
    void bwdInit() override;
    void reinit() override;
    void initVisual() override;

    void preDrawScene(core::visual::VisualParams* vp) override;
    bool drawScene(core::visual::VisualParams* vp) override;
    void postDrawScene(core::visual::VisualParams* vp) override;

    void draw(const core::visual::VisualParams* vparams) override;
    void fwdDraw(core::visual::VisualParams*) override;
    void bwdDraw(core::visual::VisualParams*) override;

    ///Register a light into the LightManager
    void putLight(Light::SPtr light);

    ///Register a vector of lights into the LightManager
    void putLights(std::vector<Light::SPtr> m_lights);

    ///Remove all lights of the LightManager
    void clear();

    void restoreDefaultLight(core::visual::VisualParams* vparams);

    void handleEvent(sofa::core::objectmodel::Event* event) override;
};

} // namespace sofa::gl::component::shader
