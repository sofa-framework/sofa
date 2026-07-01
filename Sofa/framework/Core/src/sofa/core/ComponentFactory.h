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
#pragma once

#include <sofa/config.h>
#include <sofa/core/config.h>
#include <sofa/core/ComponentCreator.h>
#include <sofa/core/ComponentRegistrationData.h>

namespace sofa::core
{

class SOFA_CORE_API ComponentFactory
{
public:

    using SOFA_CORE_DEPRECATED_OBJECTFACTORY_CLASSENTRY() ClassEntry = ComponentRegistrationData;

    using Registry = std::vector<ComponentRegistrationData::SPtr>;
    const Registry& getRegistry() const { return m_registry; }

    bool registerObjectsFromPlugin(const std::string& pluginName);

    void registerComponent(const ComponentRegistrationData::SPtr& componentRegistrationData);

    ComponentRegistrationData::SPtr findComponent(objectmodel::BaseObjectDescription* arg);

    /// Create a component given a context and a description.
    objectmodel::BaseComponent::SPtr createComponent(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg);

    /// Test if a creator exists for a given classname
    bool hasCreator(const std::string& classname) const;

    /// Fill the given vector with the registered classes from a given target
    void getEntriesFromTarget(std::vector<ComponentRegistrationData::SPtr>& result, const std::string& target) const;

    /// Return the list of classes from a given target
    std::string listClassesFromTarget(std::string target, std::string separator = ", ");

    /// Fill the given vector with all the registered classes derived from BaseClass
    template<class BaseClass>
    std::vector<ComponentRegistrationData::SPtr> getEntriesDerivedFrom() const;

    /// Return the list of classes derived from BaseClass as a string
    template<class BaseClass>
    std::string listClassesDerivedFrom(const std::string& separator = ", ") const;


    objectmodel::BaseComponent::SPtr SOFA_CORE_DEPRECATED_OBJECTFACTORY_CREATEOBJECT() createObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg);

    SOFA_CORE_DEPRECATED_OBJECTFACTORY_REGISTEROBJECTS()
    bool registerObjects(LegacyComponentRegistrationData& ro);

    SOFA_ATTRIBUTE_DEPRECATED("v26.12", "v27.12", "Use MainComponentFactory::getInstance instead")
    static ComponentFactory* getInstance();

    SOFA_ATTRIBUTE_DEPRECATED("v26.12", "v27.12", "Use MainComponentFactory::CreateComponent instead")
    static objectmodel::BaseComponent::SPtr CreateObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg);

    SOFA_ATTRIBUTE_DEPRECATED("v26.12", "v27.12", "Use MainComponentFactory::HasCreator instead")
    static bool  HasCreator(const std::string& classname);


    void getEntry(std::string) = delete;
    void getAllEntries(std::vector<ClassEntry::SPtr>& result, bool filterUnloadedPlugins = true) = delete;

    void dumpXML(std::ostream& out = std::cout) = delete;
    void dumpHTML(std::ostream& out = std::cout) = delete;

    static std::string ShortName(std::string classname) = delete;
    static void ResetAlias(std::string name, ClassEntry::SPtr previous) = delete;
    static bool AddAlias(std::string name, std::string result, bool force=false,
                         ClassEntry::SPtr* previous = nullptr) = delete;

protected:

    /// Keep track of plugins who already registered
    using RegisteredPluginSet = std::set<std::string>;
    RegisteredPluginSet m_registeredPluginSet;

    Registry m_registry;
};


template<class Class>
std::vector<ComponentRegistrationData::SPtr> ComponentFactory::getEntriesDerivedFrom() const
{
    std::vector<ComponentRegistrationData::SPtr> result;

    auto* componentClass = Class::GetClass();
    if (!componentClass)
    {
        return result;
    }

    for (const auto& component : m_registry)
    {
        if (auto* componentClassInRegistry = component->classData)
        {
            if (componentClassInRegistry->hasParent(componentClass))
            {
                result.push_back(component);
            }
        }
    }

    return result;
}

template<class BaseClass>
std::string ComponentFactory::listClassesDerivedFrom(const std::string& separator) const
{
    auto entries = getEntriesDerivedFrom<BaseClass>();

    return sofa::helper::join(entries.begin(), entries.end(),
        [](const ComponentRegistrationData::SPtr& entry){ return entry->componentName;}, separator);
}




struct SOFA_CORE_API MainComponentFactory
{
    static ComponentFactory* getInstance();

    static objectmodel::BaseComponent::SPtr CreateComponent(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg);

    // to deprecate
    static objectmodel::BaseComponent::SPtr CreateObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg);

    /// Test if a creator exists for a given classname
    static bool HasCreator(const std::string& classname);
};


template<class RealObject>
using ObjectCreator = DeprecatedAndRemoved;

using RegisterObject = DeprecatedAndRemoved;

}
