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

/**
 * @brief Factory class for SOFA components.
 *
 * This class manages the registration and instantiation of SOFA components.
 * It maintains a registry of ComponentRegistrationData which contains the creators
 * and metadata for each component.
 */
class SOFA_CORE_API ComponentFactory
{
public:

    using SOFA_CORE_DEPRECATED_OBJECTFACTORY_CLASSENTRY() ClassEntry = ComponentRegistrationData;

    using Registry = std::vector<ComponentRegistrationData::SPtr>;
    /** @brief Get the list of all registered components. */
    const Registry& getRegistry() const { return m_registry; }

    /**
     * @brief Register components defined in a plugin.
     * @param pluginName The name of the plugin to register.
     * @return true if the registration was successful or already done, false if the plugin is not loaded.
     */
    bool registerObjectsFromPlugin(const std::string& pluginName);

    /**
     * @brief Register a component in the factory.
     * @param componentRegistrationData The data containing component info and its creator.
     */
    void registerComponent(const ComponentRegistrationData::SPtr& componentRegistrationData);

    /**
     * @brief Find the most suitable component registration data for a given description.
     *
     * This method handles plugin auto-loading, alias resolution, and candidate selection
     * based on template attributes and deduction rules.
     *
     * @param context The context in which the search is performed.
     * @param arg The description of the object to find (must contain a "type" attribute).
     * @return The matching ComponentRegistrationData, or nullptr if none found.
     */
    ComponentRegistrationData::SPtr findComponent(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg);

    /**
     * @brief Create a component given a context and a description.
     *
     * This method first finds the appropriate component using @ref findComponent,
     * then instantiates it, adds it to the context, and parses its attributes.
     *
     * @param context The context where the component will be added.
     * @param arg The description of the component to create.
     * @return A smart pointer to the created component, or nullptr on failure.
     */
    objectmodel::BaseComponent::SPtr createComponent(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg);

    /** @brief Test if a creator exists for a given classname. */
    bool hasCreator(const std::string& classname) const;

    /** @brief Fill the given vector with the registered classes from a given target module/plugin. */
    void getEntriesFromTarget(std::vector<ComponentRegistrationData::SPtr>& result, const std::string& target) const;

    /** @brief Return a string list of classes from a given target module/plugin. */
    std::string listClassesFromTarget(std::string target, std::string separator = ", ");

    /** @brief Fill the given vector with all the registered classes derived from BaseClass. */
    template<class BaseClass>
    std::vector<ComponentRegistrationData::SPtr> getEntriesDerivedFrom() const;

    /** @brief Return the list of classes derived from BaseClass as a string. */
    template<class BaseClass>
    std::string listClassesDerivedFrom(const std::string& separator = ", ") const;

    objectmodel::BaseComponent::SPtr SOFA_CORE_DEPRECATED_OBJECTFACTORY_CREATEOBJECT() createObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg);

    /**
     * @brief Legacy method to register multiple objects.
     * @deprecated Use registerComponent instead.
     */
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
