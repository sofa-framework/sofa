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
// C++ Interface: LightManager
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_LIGHTMANAGER_H
#define SOFA_COMPONENT_LIGHTMANAGER_H
#include "config.h"

#include <sofa/defaulttype/SolidTypes.h>
#include <SofaOpenglVisual/Light.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/Mat.h>

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
    std::vector<Light::SPtr> lights;
    std::vector<defaulttype::Mat4x4f> lightModelViewMatrix;
    std::vector<defaulttype::Mat4x4f> lightProjectionMatrix;
    std::vector<unsigned short> m_mapShadowTextureUnit;
#ifdef SOFA_HAVE_GLEW
    //OglShadowShader* shadowShader;
    helper::vector<OglShadowShader::SPtr> shadowShaders;
#endif
    void makeShadowMatrix(unsigned int i);

public:
#ifndef __APPLE__
    enum { MAX_NUMBER_OF_LIGHTS = /*GL_MAX_LIGHTS*/ 5 };
#else
    enum { MAX_NUMBER_OF_LIGHTS = /*GL_MAX_LIGHTS*/ 2 };
#endif

    Data<bool> shadowsEnabled;
    Data<bool> softShadowsEnabled;
    Data<defaulttype::Vec4f> ambient;

protected:
    LightManager();
    virtual ~LightManager();
public:
    void init();
    void bwdInit();
    void reinit();
    void initVisual();

    void preDrawScene(core::visual::VisualParams* vp);
    bool drawScene(core::visual::VisualParams* vp);
    void postDrawScene(core::visual::VisualParams* vp);


    void draw(const core::visual::VisualParams* vparams);
    void fwdDraw(core::visual::VisualParams*);
    void bwdDraw(core::visual::VisualParams*);

    ///Register a light into the LightManager
    void putLight(Light::SPtr light);

    ///Register a vector of lights into the LightManager
    void putLights(std::vector<Light::SPtr> lights);

    ///Remove all lights of the LightManager
    void clear();

    void restoreDefaultLight(core::visual::VisualParams* vparams);

    void handleEvent(sofa::core::objectmodel::Event* event);

};

}//namespace visualmodel

}//namespace component

}//namespace sofa

#endif //SOFA_COMPONENT_LIGHT_MANAGER_H
