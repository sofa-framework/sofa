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
#include <sofa/component/linearsystem/init.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component::linearsystem
{

extern void registerCompositeLinearSystem(sofa::core::ObjectFactory* factory);
extern void registerConstantSparsityPatternSystem(sofa::core::ObjectFactory* factory);
extern void registerConstantSparsityProjectionMethod(sofa::core::ObjectFactory* factory);
extern void registerMatrixLinearSystem(sofa::core::ObjectFactory* factory);
extern void registerMatrixProjectionMethod(sofa::core::ObjectFactory* factory);

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

extern "C" {
    SOFA_COMPONENT_LINEARSYSTEM_API void initExternalModule();
    SOFA_COMPONENT_LINEARSYSTEM_API const char* getModuleName();
    SOFA_COMPONENT_LINEARSYSTEM_API const char* getModuleVersion();
    SOFA_COMPONENT_LINEARSYSTEM_API const char* getModuleLicense();
    SOFA_COMPONENT_LINEARSYSTEM_API const char* getModuleDescription();
    SOFA_COMPONENT_LINEARSYSTEM_API void registerObjects(sofa::core::ObjectFactory* factory);
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

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin defines a linear system and provides components able to assemble one from the scene.";
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    registerCompositeLinearSystem(factory);
    registerConstantSparsityPatternSystem(factory);
    registerConstantSparsityProjectionMethod(factory);
    registerMatrixLinearSystem(factory);
    registerMatrixProjectionMethod(factory);
}

} // namespace sofa::component::linearsystem
