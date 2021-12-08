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
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;
using sofa::core::ObjectFactory ;

#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileSystem.h>
using sofa::core::objectmodel::ComponentState ;

#include <SofaBaseUtils/ImportComponent.h>

#include <sofa/helper/StringUtils.h>

using std::string;

namespace sofa::component
{

ImportComponent::ImportComponent() :
    d_plugin(initData(&d_plugin, "fromPlugin", ""))
  ,d_importOldName(initData(&d_importOldName, "component(s)", ""))
  ,d_asNewName(initData(&d_asNewName, "as", ""))
{
    d_componentState.setValue(ComponentState::Invalid) ;
    addAlias(&d_importOldName, "component");
    addAlias(&d_importOldName, "components");
}

void ImportComponent::parse ( core::objectmodel::BaseObjectDescription* arg )
{
    Inherit1::parse(arg);
    const char* alias=arg->getAttribute("as");
    const char* plugin=arg->getAttribute("fromPlugin");
    const char* object=arg->getAttribute("component");
    const char* objects=arg->getAttribute("components");

    // Check that there is a plugin name.
    if(plugin==nullptr)
    {
        msg_error() << "The mandatory 'fromPlugin' attribute is missing.  "
                       "To remove this error message you need to add a 'fromPlugin' attribute pointing to a valid plugin name.";
        return ;
    }
    setName(plugin);

    // Check that there is either multiple object (and no has)
    if(objects != nullptr)
    {
        if(object != nullptr)
        {
            msg_error() << "Use of components and component at the same time is not possible.  "
                           "To remove this error message use only 'components' to load multiple ones and 'component' for single one loading.";
            return ;
        }
        if(alias != nullptr)
        {
            msg_error() << "The loading of multiple components is not compatible with name aliasing."
                           "To remove this error message you need to write either <Import component='MyComponent' as='' or .";
            return ;
        }
    }

    innerInit();
}

void ImportComponent::innerInit()
{
    d_componentState.setValue(ComponentState::Invalid);
    const std::string plugin = d_plugin.getValue();

    // First load the plugin if it is not yet done
    type::vector< std::string > failed;
    const std::string name = sofa::helper::system::FileSystem::cleanPath( plugin );
    auto& pluginManager = sofa::helper::system::PluginManager::getInstance();
    if ( !pluginManager.pluginIsLoaded(name) )
    {
        std::ostringstream errmsg;
        if(!pluginManager.loadPlugin(name, sofa::helper::system::PluginManager::getDefaultSuffix(), true, true, &errmsg))
        {
            msg_error() << errmsg.str();
        }
    }

    std::vector<std::string> componentsAlias;
    std::string cleanedString=d_importOldName.getValue();
    sofa::helper::replaceAll(cleanedString, " ", "");
    std::vector<std::string> componentsName = sofa::helper::split(cleanedString, ',');
    if(componentsName.empty())
    {
        msg_error() << "No component to import, please set some";
        return;
    }
    if(componentsName.size() != 1 && d_asNewName.isSet())
    {
        msg_error() << "Cannot import multiple components with aliases. Remove the 'as' attribute.";
    }

    // Only one componentName, so either a real name or the * operator
    if(componentsName.size() == 1)
    {
        // Let's first handle the case with '*'
        if(componentsName[0]=="*")
        {
            componentsName.clear();
            std::vector<sofa::core::ObjectFactory::ClassEntry::SPtr> entries;
            ObjectFactory::getInstance()->getEntriesFromTarget(entries, std::string(plugin));

            // For each entry create the full name out of the plugin's name and the target's one.
            for(auto& entry : entries)
            {
                const std::string fullname = std::string(plugin)+"."+entry->className;
                if(entry->compilationTarget == plugin)
                {
                    componentsName.push_back(entry->className);
                }
            }
        }
        else if(d_asNewName.isSet())
        {
            componentsAlias.push_back(d_asNewName.getValue());
        }
    }

    // Properly initialized
    if(componentsAlias.size() != componentsName.size())
        componentsAlias = componentsName ;

    // If we arrive here, this means there are more than one component's name,
    // for each of them we need to create the fully qualified name (the one used in the factory)
    // and map it to the requested alias.
    std::stringstream infoMessage;
    infoMessage << "Loading components: " << componentsName.size() <<  msgendl;
    for(unsigned int i=0;i<componentsName.size();++i)
    {
        const auto& componentOldName = componentsName[i];
        const auto& componentNewName = componentsAlias[i];

        std::stringstream tmp;
        tmp << plugin << "." << componentOldName;
        std::string fullname = tmp.str();

        infoMessage << " - " << componentOldName << " as " << componentNewName << msgendl;

        // Check there is a real component with that name in the factory
        if(!ObjectFactory::getInstance()->hasCreator(fullname))
        {
            std::stringstream errorMessage;
            std::vector<sofa::core::ObjectFactory::ClassEntry::SPtr> entries;
            ObjectFactory::getInstance()->getEntriesFromTarget(entries, std::string(plugin));
            errorMessage << "The component '"<< componentOldName << "' does not correspond to a valid one available in the plugin '"<< plugin <<"'." << msgendl;
            errorMessage << "Content of " << d_plugin.getValueString() << ":" << msgendl;
            for(auto& entry : entries)
            {
                errorMessage << " - " << entry->className << msgendl;
            }
            errorMessage << "To remove this error message you need to fix your scene and provide a valid component ClassName in the 'import' attribute. ";
            msg_error() << errorMessage.str();
            return ;
        }

        // Register the alias
        ObjectFactory::getInstance()->addAlias(componentNewName, fullname, true);
    }
    msg_info() << infoMessage.str();

    d_componentState.setValue(ComponentState::Valid);
}

int ImportComponentClass = RegisterObject("Loads a plugin and import its content into the current namespace. ")
        .add< ImportComponent >()
        .addTargetName(sofa_tostring(SOFA_TARGET));

} // namespace sofa::component
