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
#include <sofa/core/ComponentFactory.h>
#include <sofa/defaulttype/TemplatesAliases.h>
#include <sofa/helper/ComponentChange.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/url.h>

namespace sofa::core
{

std::vector<ComponentDescription::SPtr> ComponentFactory::getComponentsFromName(const std::string& componentName) const
{
    std::vector<ComponentDescription::SPtr> result;

    std::string componentToSearch = componentName;

    using sofa::helper::lifecycle::renamedComponents;
    auto renamedComponent = renamedComponents.find(componentName);
    if( renamedComponent != renamedComponents.end() )
    {
        componentToSearch = renamedComponent->second.getNewName();
    }

    for (const auto& component : m_registry)
    {
        const auto fullName = component->componentModule + "." + component->componentName;

        if (component->componentName == componentToSearch || fullName == componentToSearch)
        {
            result.push_back(component);
        }
        else
        {
            for (const auto& alias : component->aliases)
            {
                const auto fullNameAlias = component->componentModule + "." + alias;

                if (alias == componentToSearch || fullNameAlias == componentToSearch)
                {
                    result.push_back(component);
                }
            }
        }
    }

    return result;
}

typedef struct ObjectRegistrationEntry
{
    inline static const char* symbol = "registerObjects";
    typedef void (*FuncPtr) (sofa::core::ComponentFactory*);
    FuncPtr func;
    void operator()(sofa::core::ComponentFactory* data) const
    {
        if (func) return func(data);
    }
    ObjectRegistrationEntry() :func(nullptr) {}
} ObjectRegistrationEntry;

bool ComponentFactory::registerObjectsFromPlugin(const std::string& pluginName)
{
    sofa::helper::system::PluginManager& pluginManager = sofa::helper::system::PluginManager::getInstance();
    auto* plugin = pluginManager.getPlugin(pluginName);
    if (plugin == nullptr)
    {
        msg_error("ObjectFactory") << pluginName << " has not been loaded yet.";
        return false;
    }

    // do not register if it was already done before
    if(m_registeredPluginSet.contains(pluginName))
    {
        // This warning should be generalized (i.e not only in dev mode) when runSofa will not auto-load modules/plugins by default anymore
        // Commented warning since it is triggered even for SOFA meta-modules (e.g. Sofa.Components)
        // dmsg_warning("ObjectFactory") << pluginName << " has already registered its components.";
        return false;
    }

    ObjectRegistrationEntry registerObjects;
    if (pluginManager.getEntryFromPlugin(plugin, registerObjects))
    {
        registerObjects(this);
        m_registeredPluginSet.insert(pluginName);
        return true;
    }
    else
    {
        return false;
    }
}

bool ComponentFactory::registerObjects(LegacyComponentRegistrationData& ro)
{
    // auto& creators = ro.creators;
    //
    // if (creators.empty())
    // {
    //     msg_error() << "No creator provided";
    //     return false;
    // }
    //
    // for (auto& creator : ro.creators)
    // {
    //     ComponentDescription::SPtr component = std::make_shared<ComponentDescription>();
    //
    //     component->componentName = ro.componentName;
    //     component->aliases = ro.aliases;
    //     component->componentNamespace = ro.componentNamespace;
    //     component->componentModule = ro.componentModule;
    //
    //     component->description = ro.description;
    //     component->authors = sofa::helper::join(ro.authors, ",");
    //     component->license = ro.license;
    //     component->documentationURL = ro.documentationURL;
    //
    //     {
    //         //special cases for official documentation
    //         const auto modulePaths = sofa::helper::split(component->componentModule, '.');
    //         if (modulePaths.size() > 2 && modulePaths[0] == "Sofa" && modulePaths[1] == "Component")
    //         {
    //             std::string officialDocURL = std::string(sofa::SOFA_DOCUMENTATION_URL) + std::string("components/");
    //             officialDocURL += sofa::helper::join(modulePaths.begin() + 2, modulePaths.end(),
    //                 [](const std::string& m){ return sofa::helper::downcaseString(m);}, "/");
    //             officialDocURL += std::string("/") + sofa::helper::downcaseString(component->componentName);
    //
    //             component->documentationURL.insert(officialDocURL);
    //         }
    //     }
    //
    //     component->creator = std::move(creator);
    //
    //     this->m_registry.push_back(component);
    // }

    return true;

}

namespace
{
void autoLoadPluginIfNameContainsPluginName(ComponentFactory& self, const std::string& classname)
{
    // The last dot separates the module name from the component name
    // Example: Module.Name.ComponentName (it is common to have dots in the module names)
    // It is assumed that the component name does not contain any dot
    auto lastDot = classname.find_last_of('.');
    if (lastDot != std::string::npos)
    {
        const auto pluginName = classname.substr(0, lastDot);

        const auto [path, loaded] = helper::system::PluginManager::getInstance().isPluginLoaded(pluginName);
        if (!loaded)
        {
            auto status = helper::system::PluginManager::getInstance().loadPluginByName(pluginName);
            if (status == helper::system::PluginManager::PluginLoadStatus::SUCCESS)
            {
                self.registerObjectsFromPlugin(pluginName);
            }
        }
    }
}

std::vector<ComponentDescription::SPtr> selectCandidates(const std::vector<ComponentDescription::SPtr>& candidates, objectmodel::BaseObjectDescription* arg)
{
    std::vector<ComponentDescription::SPtr> matchingCandidates;

    for (const auto& candidate : candidates)
    {
        bool matchAllTemplateParameters = true;
        for (const auto& [attribute, value] : candidate->templateAttributes)
        {
            const char* attr = arg->getAttribute(attribute, nullptr);
            if (attr == nullptr)
            {
                matchAllTemplateParameters = false;
            }
            else
            {
                const std::string attrStr { attr };
                if (defaulttype::TemplateAliases::resolveAlias(attrStr) != value)
                {
                    matchAllTemplateParameters = false;
                }
            }
        }

        if (matchAllTemplateParameters)
        {
            matchingCandidates.push_back(candidate);
        }
    }

    return matchingCandidates;
}

auto createComponentFrom(const ComponentDescription::SPtr& desc, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    auto component = desc->creator->create();

    if (component)
    {
        if (context)
        {
            context->addObject(component);
        }

        component->parse(arg);
    }

    return component;
}

}

objectmodel::BaseComponent::SPtr ComponentFactory::createComponent(
    objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    if (!arg)
        return nullptr;

    const char* typeAttribute = arg->getAttribute( "type", nullptr);

    if (typeAttribute == nullptr)
        return nullptr;

    std::string classname {typeAttribute};
    autoLoadPluginIfNameContainsPluginName(*this, classname);

    auto candidates = this->getComponentsFromName(classname);

    if (candidates.empty())
    {
        return nullptr;
    }

    std::erase_if(candidates, [](const ComponentDescription::SPtr& candidate)
    {
        return helper::system::PluginManager::getInstance().isPluginUnloaded(candidate->componentModule);
    });

    const auto matchingTemplates = selectCandidates(candidates, arg);

    if (!matchingTemplates.empty())
    {
        msg_warning_when(matchingTemplates.size() > 1) << "Multiple candidates with the same templates";
        return createComponentFrom(matchingTemplates.front(), context, arg);
    }

    //todo: template deduction

    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a->instantiationPriority > b->instantiationPriority; });


    return createComponentFrom(candidates.front(), context, arg);
}

bool ComponentFactory::hasCreator(const std::string& classname) const
{
    return std::any_of(m_registry.begin(), m_registry.end(),
        [&](const auto& component){ return component->componentName == classname; });
}

void ComponentFactory::getEntriesFromTarget(std::vector<ComponentDescription::SPtr>& result,
                                            const std::string& target) const
{
    for (const auto& component : m_registry)
    {
        if (component->componentModule == target)
        {
            result.push_back(component);
        }
    }
}

std::string ComponentFactory::listClassesFromTarget(std::string target, std::string separator)
{
    std::vector<ClassEntry::SPtr> entries;
    this->getEntriesFromTarget(entries, target);
    return sofa::helper::join(entries, separator);
}

ComponentFactory* ComponentFactory::getInstance() { return MainComponentFactory::getInstance(); }
objectmodel::BaseComponent::SPtr ComponentFactory::CreateObject(
    objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    return MainComponentFactory::CreateComponent(context, arg);
}

bool ComponentFactory::HasCreator(const std::string& classname)
{
    return MainComponentFactory::HasCreator(classname);
}

bool MainComponentFactory::HasCreator(const std::string& classname)
{
    return getInstance()->hasCreator(classname);
}

}  // namespace sofa::core
