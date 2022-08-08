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
#include <SofaMiscExtra/initSofaMiscExtra.h>

#include <sofa/helper/system/PluginManager.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;

namespace sofa::component
{

void initSofaMiscExtra()
{
    static bool first = true;
    if (first)
    {
        msg_deprecated("SofaMiscExtra") << "SofaMiscExtra is deprecated. It will be removed at v23.06. Use Sofa.Component.Engine.Generate instead.";

        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.Engine.Generate");

        first = false;
    }
}

extern "C" {
    SOFA_SOFAMISCEXTRA_API void initExternalModule();
    SOFA_SOFAMISCEXTRA_API const char* getModuleName();
    SOFA_SOFAMISCEXTRA_API const char* getModuleVersion();
    SOFA_SOFAMISCEXTRA_API const char* getModuleLicense();
    SOFA_SOFAMISCEXTRA_API const char* getModuleDescription();
    SOFA_SOFAMISCEXTRA_API const char* getModuleComponentList();
}

void initExternalModule()
{
    initSofaMiscExtra();
}

const char* getModuleName()
{
    return sofa_tostring(SOFA_TARGET);
}

const char* getModuleVersion()
{
    return sofa_tostring(SOFAMISCEXTRA_VERSION);
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin contains contains features about Misc Extra.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

} // namespace sofa::component
