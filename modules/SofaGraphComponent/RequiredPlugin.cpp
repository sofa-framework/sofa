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

int RequiredPluginClass = core::RegisterObject("Load the required plugins")
        .add< RequiredPlugin >();

RequiredPlugin::RequiredPlugin()
    : d_pluginName( initData(&d_pluginName, "pluginName", "plugin name (or several names if you need to load different plugins or a plugin with several alternate names)"))
    , d_suffixMap ( initData(&d_suffixMap , "suffixMap", "standard->custom suffixes pairs (to be used if the plugin is compiled outside of Sofa with a non standard way of differenciating versions), using ! to represent empty suffix"))
    , d_stopAfterFirstNameFound( initData(&d_stopAfterFirstNameFound , false, "stopAfterFirstNameFound", "Stop after the first plugin name that is loaded successfully"))
    , d_stopAfterFirstSuffixFound( initData(&d_stopAfterFirstSuffixFound , true, "stopAfterFirstSuffixFound", "For each plugin name, stop after the first suffix that is loaded successfully"))
    , d_requireOne ( initData(&d_requireOne , false, "requireOne", "Display an error message if no plugin names were successfully loaded"))
    , d_requireAll ( initData(&d_requireAll , true, "requireAll", "Display an error message if any plugin names failed to be loaded"))
{
    this->f_printLog.setValue(true); // print log by default, to identify which pluging is responsible in case of a crash during loading
}

void RequiredPlugin::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    Inherit1::parse(arg);
    loadPlugin();
}

void RequiredPlugin::loadPlugin()
{
    sofa::helper::system::PluginManager* pluginManager = &sofa::helper::system::PluginManager::getInstance();
    std::string defaultSuffix = pluginManager->getDefaultSuffix();
    const helper::vector<helper::fixed_array<std::string,2> >& sMap = d_suffixMap.getValue();
    helper::vector<std::string> suffixVec;
    if (!sMap.empty())
    {
        std::string skey = (defaultSuffix.empty() ? std::string("!") : defaultSuffix);
        for (std::size_t i = 0; i < sMap.size(); ++i)
        {
            if (sMap[i][0] == skey)
            {
                suffixVec.push_back(sMap[i][1] == std::string("!") ? std::string(""):sMap[i][1]);
            }
        }
    }
    if (suffixVec.empty())
        suffixVec.push_back(defaultSuffix);

    /// In case the pluginName is not set we copy the provided name into the set to load.
    if(!d_pluginName.isSet() && name.isSet())
    {
        helper::WriteOnlyAccessor<Data<helper::vector<std::string>>> pluginsName = d_pluginName ;
        pluginsName.push_back(this->getName());
    }

    const helper::vector<std::string>& nameVec = d_pluginName.getValue();
    helper::vector<std::string> nameVecCopy=nameVec;

    helper::vector< std::string > loaded;
    helper::vector< std::string > failed;
    std::ostringstream errmsg;
    for (std::size_t nameIndex = 0; nameIndex < nameVecCopy.size(); ++nameIndex)
    {
        const std::string& name = nameVecCopy[nameIndex];
        //sout << "Loading " << name << sendl;
        bool nameLoaded = false;
        for (std::size_t suffixIndex = 0; suffixIndex < suffixVec.size(); ++suffixIndex)
        {
            const std::string& suffix = suffixVec[suffixIndex];
            std::string pluginPath = pluginManager->findPlugin(name, suffix, false);
            bool result = !pluginPath.empty();
            if (result && !pluginManager->pluginIsLoaded(pluginPath))
            {
                result = pluginManager->loadPlugin(pluginPath, suffix, false, &errmsg);
            }
            if (result)
            {
                msg_info("RequiredPlugin") << "Loaded " << pluginPath;
                loaded.push_back(pluginPath);
                nameLoaded = true;
                if (d_stopAfterFirstSuffixFound.getValue()) break;
            }
        }
        if (!nameLoaded)
        {
            failed.push_back(name);
        }
        else
        {
            if (d_stopAfterFirstNameFound.getValue()) break;
        }
    }
    if (!failed.empty())
    {
        if ((d_requireAll.getValue() || (d_requireOne.getValue() && loaded.empty())))
        {
            msg_error("RequiredPlugin") << errmsg.str();
            msg_error("RequiredPlugin") <<(failed.size()>1?"s":"")<<" failed to load: " << failed ;
        }
        else
        {
            msg_warning("RequiredPlugin") << errmsg.str();
            msg_warning("RequiredPlugin") << "Optional/alternate plugin"<<(failed.size()>1?"s":"")<<" failed to load: " << failed;
        }
    }
    pluginManager->init();

}

}

}

}
