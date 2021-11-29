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

#include <SofaBaseUtils/FromComponent.h>

using std::string;

namespace sofa::component
{

FromComponent::FromComponent() :
    d_plugin(initData(&d_plugin, "plugin", ""))
  ,d_import_old_name(initData(&d_import_old_name, "import", ""))
  ,d_as_new_name(initData(&d_as_new_name, "as", ""))
{
    d_componentState.setValue(ComponentState::Invalid) ;
}

void FromComponent::parse ( core::objectmodel::BaseObjectDescription* arg )
{
    BaseObject::parse(arg) ;

    const char* plugin=arg->getAttribute("plugin") ;
    const char* object=arg->getAttribute("import") ;
    const char* alias=arg->getAttribute("as") ;

    if(plugin==nullptr)
    {
        msg_error() << "The mandatory 'plugin' attribute is missing.  "
                           "The component is disabled.  "
                           "To remove this error message you need to add a targetcomponent attribute pointing to a valid component's ClassName.";
        return ;
    }
    string starget(plugin) ;

    if(object==nullptr)
    {
        msg_error() << "The mandatory 'import' attribute is missing.  "
                           "The component is disabled.  "
                           "To remove this error message you need to add an alias attribute with a valid string component's ClassName.";
        return ;
    }

    // First load the plugin if it is not yet done
    type::vector< std::string > failed;
    std::ostringstream errmsg;
    const std::string name = sofa::helper::system::FileSystem::cleanPath( plugin );
    auto& pluginManager = sofa::helper::system::PluginManager::getInstance();
    if ( !pluginManager.pluginIsLoaded(name) )
    {
        if(!pluginManager.loadPlugin(name, sofa::helper::system::PluginManager::getDefaultSuffix(), true, true, &errmsg))
        {
            msg_error() << errmsg.str();
        }
    }


    // Then import the requested names.
    if(std::string(object)=="*")
    {
        std::vector<sofa::core::ObjectFactory::ClassEntry::SPtr> entries;
        ObjectFactory::getInstance()->getEntriesFromTarget(entries, std::string(plugin));
        for(auto& entry : entries)
        {
            string fullname = std::string(plugin)+"."+entry->className;
            ObjectFactory::getInstance()->addAlias(entry->className, fullname,true);
        }
        if(alias != nullptr)
        {
            msg_error() << "using 'as' and '*' at the same time is not supposed to work.";
            return ;
        }
        d_componentState.setValue(ComponentState::Valid);
        return;
    }

    if(alias == nullptr)
    {
        alias = object;
    }

    std::stringstream tmp;
    tmp << plugin << "." << object;
    std::string fullname = tmp.str();
    if(!ObjectFactory::getInstance()->hasCreator(fullname))
    {
        std::stringstream tmp;
        std::vector<sofa::core::ObjectFactory::ClassEntry::SPtr> entries;
        ObjectFactory::getInstance()->getEntriesFromTarget(entries, std::string(plugin));
        tmp << "The object '"<< object << "' does not correspond to a valid component available in plugin '"<< plugin <<"'." << msgendl;
        tmp << "Content of " << d_plugin.getValueString() << ":" << msgendl;
        for(auto& entry : entries)
        {
            tmp << " - " << entry->className << msgendl;
        }
        tmp << "To remove this error message you need to fix your scene and provide a valid component ClassName in the 'import' attribute. ";
        msg_error() << tmp.str();
        return ;
    }
    ObjectFactory::getInstance()->addAlias(alias, fullname,true);
    d_componentState.setValue(ComponentState::Valid) ;
}

int FromComponentClass = RegisterObject("This object create an alias to a component name to make the scene more readable. ")
        .add< FromComponent >()
        .addTargetName(sofa_tostring(SOFA_TARGET))
        .addAlias("From", false)
        ;

} // namespace sofa::component
