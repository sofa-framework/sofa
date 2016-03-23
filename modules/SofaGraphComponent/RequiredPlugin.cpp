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
    : pluginName( initData(&pluginName, "pluginName", "Name of the plugin to loaded. If this is empty, the name of this component is used as plugin name."))
{
    this->f_printLog.setValue(true); // print log by default, to identify which pluging is responsible in case of a crash during loading
}

void RequiredPlugin::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    Inherit1::parse(arg);
    if (!pluginName.getValue().empty() || !name.getValue().empty())
        loadPlugin();
}

void RequiredPlugin::loadPlugin()
{
    if(pluginName.getValue().empty()) pluginName.setValue( name.getValue() );

    PluginManager& pluginManager = PluginManager::getInstance();

    const std::string path = pluginManager.findPlugin(pluginName.getValue());
    if (path != "")
    {
        if (!PluginManager::getInstance().pluginIsLoaded(path))
        {
            if (PluginManager::getInstance().loadPlugin(path))
            {
                const std::string guiPath = pluginManager.findPlugin(pluginName.getValue() + "_" + PluginManager::s_gui_postfix);
                if (guiPath != "")
                {
                    PluginManager::getInstance().loadPlugin(guiPath);
                }
            }
        }
    }
    else
    {
        const std::string msg = "Plugin not found: \"" + pluginName.getValue() + "\"";
        msg_error("RequiredPlugin") << msg;

    }
}

}

}

}
