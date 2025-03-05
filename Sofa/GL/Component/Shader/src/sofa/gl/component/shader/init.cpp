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
#include <sofa/gl/component/shader//init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::gl::component::shader
{

extern void registerCompositingVisualLoop(sofa::core::ObjectFactory* factory);
extern void registerDirectionalLight(sofa::core::ObjectFactory* factory);
extern void registerPositionalLight(sofa::core::ObjectFactory* factory);
extern void registerSpotlLight(sofa::core::ObjectFactory* factory);
extern void registerLightManager(sofa::core::ObjectFactory* factory);
extern void registerOglAttribute(sofa::core::ObjectFactory* factory);
extern void registerOglOITShader(sofa::core::ObjectFactory* factory);
extern void registerOglRenderingSRGB(sofa::core::ObjectFactory* factory);
extern void registerOglShader(sofa::core::ObjectFactory* factory);
extern void registerOglShaderDefineMacro(sofa::core::ObjectFactory* factory);
extern void registerOglShaderVisualModel(sofa::core::ObjectFactory* factory);
extern void registerOglShadowShader(sofa::core::ObjectFactory* factory);
extern void registerOglTexture(sofa::core::ObjectFactory* factory);
extern void registerOglTexturePointer(sofa::core::ObjectFactory* factory);
extern void registerOglVariable(sofa::core::ObjectFactory* factory);
extern void registerOrderIndependentTransparencyManager(sofa::core::ObjectFactory* factory);
extern void registerPostProcessManager(sofa::core::ObjectFactory* factory);
extern void registerVisualManagerPass(sofa::core::ObjectFactory* factory);
extern void registerVisualManagerSecondaryPass(sofa::core::ObjectFactory* factory);

extern "C" {
    SOFA_EXPORT_DYNAMIC_LIBRARY void initExternalModule();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleVersion();
    SOFA_EXPORT_DYNAMIC_LIBRARY void registerObjects(sofa::core::ObjectFactory* factory);
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    registerCompositingVisualLoop(factory);
    registerDirectionalLight(factory);
    registerPositionalLight(factory);
    registerSpotlLight(factory);
    registerLightManager(factory);
    registerOglAttribute(factory);
    registerOglOITShader(factory);
    registerOglRenderingSRGB(factory);
    registerOglShader(factory);
    registerOglShaderDefineMacro(factory);
    registerOglShaderVisualModel(factory);
    registerOglShadowShader(factory);
    registerOglTexture(factory);
    registerOglTexturePointer(factory);
    registerOglVariable(factory);
    registerOrderIndependentTransparencyManager(factory);
    registerPostProcessManager(factory);
    registerVisualManagerPass(factory);
    registerVisualManagerSecondaryPass(factory);
}

void init()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);

        first = false;
    }
}

} // namespace sofa::gl::component::shader
