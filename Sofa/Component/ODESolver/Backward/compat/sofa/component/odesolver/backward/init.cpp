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
#include <sofa/component/odesolver/backward/init.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/Modules.h>

namespace sofa::component::odesolver::backward
{
    
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
    const auto isLoaded = sofa::helper::system::PluginManager::getInstance().isPluginLoaded(Sofa.Component.IntegrationScheme.Backward);
    bool objLoded = false;

    if (isLoaded.second)
    {
        objLoaded = factory->registerObjectsFromPlugin(Sofa.Component.IntegrationScheme.Backward);

        if (!objLoaded)
            msg_info_once("Sofa.Component.ODESolver.Backward")<<"Registering objects from Sofa.Component.IntegrationScheme.Backward failed.";
    }
}

void init()
{
    msg_deprecated_once(MODULE_NAME)<<"This plugin is empty since v26.12 and will be removed in v27.12, load Sofa.Component.IntegrationScheme.Backward instead.";
    auto status = sofa::helper::system::PluginManager::getInstance().loadPluginByName(Sofa.Component.IntegrationScheme.Backward) ;

    if (status <= sofa::helper::system::PluginManager::PluginLoadStatus::ALREADY_LOADED)
        msg_info_once("Sofa.Component.ODESolver.Backward")<<"Sofa.Component.IntegrationScheme.Backward has been loaded automatically.";
    else
        msg_warning_once("Sofa.Component.ODESolver.Backward")<<"Tried to load Sofa.Component.IntegrationScheme.Backward automatically but failed.";
}

} // namespace sofa::component::odesolver::backward
