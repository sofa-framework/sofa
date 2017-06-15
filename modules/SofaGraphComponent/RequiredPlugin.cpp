/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "RequiredPlugin.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/logging/Messaging.h>

using sofa::helper::system::PluginManager;

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(RequiredPlugin)

int RequiredPluginClass = core::RegisterObject("Load required plugin")
        .add< RequiredPlugin >();

RequiredPlugin::RequiredPlugin()
    : d_pluginName( initData(&d_pluginName, "pluginName", "Name of the plugin to loaded. If this is empty, the name of this component is used as plugin name."))
{
    this->f_printLog.setValue(true); // print log by default, to identify which pluging is responsible in case of a crash during loading
}

void RequiredPlugin::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    Inherit1::parse(arg);

    const helper::vector<std::string>& pluginName = d_pluginName.getValue();

    // if no plugin name is given, try with the component name
    if(pluginName.empty())
        loadPlugin( name.getValue() );
    else
        for( const auto& it: pluginName )
            loadPlugin( it );
}

void RequiredPlugin::loadPlugin( const std::string& pluginName )
{
    PluginManager& pluginManager = PluginManager::getInstance();

    const std::string path = pluginManager.findPlugin(pluginName);
    if (path != "")
    {
        if (!PluginManager::getInstance().pluginIsLoaded(path))
        {
            if (PluginManager::getInstance().loadPlugin(path))
            {
                const std::string guiPath = pluginManager.findPlugin(pluginName + "_" + PluginManager::s_gui_postfix);
                if (guiPath != "")
                {
                    PluginManager::getInstance().loadPlugin(guiPath);
                }
            }
        }
    }
    else
    {
        msg_error("RequiredPlugin") << "Plugin not found: \"" + pluginName + "\"";
    }
}

}

}

}
