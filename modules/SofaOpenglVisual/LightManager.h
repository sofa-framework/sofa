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
#ifndef SOFA_COMPONENT_LIGHTMANAGER_H
#define SOFA_COMPONENT_LIGHTMANAGER_H
#include "config.h"

#include <sofa/defaulttype/SolidTypes.h>
#include <SofaOpenglVisual/Light.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/types/RGBAColor.h>

#ifdef SOFA_HAVE_GLEW
#include <SofaOpenglVisual/OglShadowShader.h>
#endif

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief Utility to manage lights into an Opengl scene
 *
 *  This class must be used with the Light class.
 *  It centralizes all the Lights and managed them.
 *
 */
class SOFA_OPENGL_VISUAL_API LightManager : public core::visual::VisualManager
{
public:
    SOFA_CLASS(LightManager, core::visual::VisualManager);

private:
    std::vector<Light::SPtr>          m_lights;
    std::vector<defaulttype::Mat4x4f> m_lightModelViewMatrix;
    std::vector<defaulttype::Mat4x4f> m_lightProjectionMatrix;
    std::vector<unsigned short>       m_mapShadowTextureUnit;

#ifdef SOFA_HAVE_GLEW
    //OglShadowShader* shadowShader;
    helper::vector<OglShadowShader::SPtr> m_shadowShaders;
#endif
    void makeShadowMatrix(unsigned int i);

public:
    enum { MAX_NUMBER_OF_LIGHTS = /*GL_MAX_LIGHTS*/ 8 };

    //TODO(dmarchal): sofa guidelines.
    Data<bool>                  d_shadowsEnabled; ///< Enable Shadow in the scene. (default=0)
    Data<bool>                  d_softShadowsEnabled; ///< If Shadows enabled, Enable Variance Soft Shadow in the scene. (default=0)
    Data<sofa::helper::types::RGBAColor>  d_ambient; ///< Ambient lights contribution (Vec4f)(default=[0.0f,0.0f,0.0f,0.0f])
    Data<bool>                  d_drawIsEnabled; ///< enable/disable drawing of lights shadow textures. (default=false)

protected:
    LightManager();
    virtual ~LightManager();

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

}//namespace visualmodel

}//namespace component

}//namespace sofa

#endif //SOFA_COMPONENT_LIGHT_MANAGER_H
