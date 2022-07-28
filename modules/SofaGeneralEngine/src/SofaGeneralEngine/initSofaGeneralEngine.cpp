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
#include <SofaGeneralEngine/initSofaGeneralEngine.h>

#include <sofa/helper/system/PluginManager.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;

namespace sofa::component
{

void initSofaGeneralEngine()
{
    static bool first = true;
    if (first)
    {
        msg_deprecated("SofaEngine") << "SofaGeneralEngine is deprecated. It will be removed at v23.06. Use Sofa.Component.Engine.Analyze, Sofa.Component.Engine.Generate, Sofa.Component.Engine.Select and Sofa.Component.Engine.Transform instead.";

        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.Engine.Analyze");
        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.Engine.Generate");
        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.Engine.Select");
        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.Engine.Transform");

#if SOFAGENERALENGINE_HAVE_SOFA_GL_COMPONENT_ENGINE == 1
        msg_deprecated("SofaGeneralEngine") << "Moreover, use Sofa.GL.Component.Engine if you need TextureInterpolation.";

        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.GL.Component.Engine");
#endif // SOFAGENERALENGINE_HAVE_SOFA_GL_COMPONENT_ENGINE == 1
        first = false;
    }
}

extern "C" {
    SOFA_SOFAGENERALENGINE_API void initExternalModule();
    SOFA_SOFAGENERALENGINE_API const char* getModuleName();
    SOFA_SOFAGENERALENGINE_API const char* getModuleVersion();
    SOFA_SOFAGENERALENGINE_API const char* getModuleLicense();
    SOFA_SOFAGENERALENGINE_API const char* getModuleDescription();
    SOFA_SOFAGENERALENGINE_API const char* getModuleComponentList();
}

void initExternalModule()
{
    initSofaGeneralEngine();
}

const char* getModuleName()
{
    return sofa_tostring(SOFA_TARGET);
}

const char* getModuleVersion()
{
    return sofa_tostring(SOFAGENERALENGINE_VERSION);
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin contains contains features about General Animation Loop.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

} // namespace sofa::component
