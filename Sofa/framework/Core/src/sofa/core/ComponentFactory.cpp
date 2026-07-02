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
#include <sofa/core/ComponentFactoryFilters.h>
#include <sofa/defaulttype/TemplatesAliases.h>
#include <sofa/helper/ComponentChange.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/DiffLib.h>
#include <sofa/url.h>

namespace sofa::helper::logging
{

inline bool notMuted(const core::ComponentFactory*)
{
    return true;
}

inline ComponentInfo::SPtr getComponentInfo(const core::ComponentFactory*)
{
    return std::make_shared<ComponentInfo>("ComponentFactory");
}

}

namespace sofa::core
{

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
    auto& creators = ro.m_componentCreators;

    if (creators.empty())
    {
        msg_error() << "No creator provided";
        return false;
    }

    for (std::size_t i = 0; i < creators.size(); ++i)
    {
        auto creator = creators[i];

        auto component = std::make_shared<ComponentRegistrationData>();

        if (i < ro.m_componentNames.size())
        {
            component->componentName = ro.m_componentNames[i];
        }
        component->description = ro.m_description;
        if (i < ro.m_moduleNames.size())
        {
            component->componentModule = ro.m_moduleNames[i];
        }
        if (i < ro.m_instantiationPriority.size())
        {
            component->instantiationPriority = ro.m_instantiationPriority[i];
        }
        if (i < ro.m_componentTemplates.size())
        {
            component->templateAttributes.emplace_back("template", ro.m_componentTemplates[i]);
        }

        component->creator = creator->clone();

        component->aliases = ro.m_aliases;
        component->authors.insert(ro.m_authors);
        component->license = ro.m_license;
        component->documentationURL.insert(ro.m_documentationURL);

        {
            //special cases for official documentation
            const auto modulePaths = sofa::helper::split(component->componentModule, '.');
            if (modulePaths.size() > 2 && modulePaths[0] == "Sofa" && modulePaths[1] == "Component")
            {
                std::string officialDocURL = std::string(sofa::SOFA_DOCUMENTATION_URL) + std::string("components/");
                officialDocURL += sofa::helper::join(modulePaths.begin() + 2, modulePaths.end(),
                    [](const std::string& m){ return sofa::helper::downcaseString(m);}, "/");
                officialDocURL += std::string("/") + sofa::helper::downcaseString(component->componentName);

                component->documentationURL.insert(officialDocURL);
            }
        }

        component->templateDeductionRule = ro.m_templateDeductionRules[i];

        this->m_registry.push_back(component);
    }

    return true;

}
void ComponentFactory::registerComponent(
    const ComponentRegistrationData::SPtr& componentRegistrationData)
{
    //check no duplicate
    for (const auto& component : m_registry)
    {
        if (component->componentName == componentRegistrationData->componentName)
        {
            auto allTemplateAttributes = true;
            for (const auto& templateAttribute : componentRegistrationData->templateAttributes)
            {
                if (std::find(component->templateAttributes.begin(), component->templateAttributes.end(),
                    templateAttribute) ==  component->templateAttributes.end())
                {
                    allTemplateAttributes = false;
                }
            }
            if (!componentRegistrationData->templateAttributes.empty() && allTemplateAttributes)
            {
                msg_error() << "Attempt to register a new component in the factory with identical attributes than " << *component;
            }
        }
    }

    m_registry.push_back(componentRegistrationData);
}

namespace
{


std::vector<ComponentRegistrationData::SPtr> getComponentsFromName(
    const ComponentFactory& self,
    const std::string& componentName,
    const std::string& pluginName)
{
    std::vector<ComponentRegistrationData::SPtr> result;

    std::string componentToSearch = componentName;

    using sofa::helper::lifecycle::renamedComponents;
    auto renamedComponent = renamedComponents.find(componentName);
    if( renamedComponent != renamedComponents.end() )
    {
        componentToSearch = renamedComponent->second.getNewName();
    }

    for (const auto& component : self.getRegistry())
    {
        if (!pluginName.empty() && component->componentModule != pluginName)
        {
            continue;
        }

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

void extractModuleName(const std::string& inputClassName, std::string& className, std::string& moduleName)
{
    // The last dot separates the module name from the component name
    // Example: Module.Name.ComponentName (it is common to have dots in the module names)
    // It is assumed that the component name does not contain any dot
    auto lastDot = inputClassName.find_last_of('.');
    if (lastDot != std::string::npos)
    {
        moduleName = inputClassName.substr(0, lastDot);
        className = inputClassName.substr(lastDot + 1);
    }
    else
    {
        moduleName = {};
        className = inputClassName;
    }
}

void autoLoadPlugin(ComponentFactory& self, const std::string& pluginName)
{
    if (!pluginName.empty())
    {
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

std::vector<std::string> similarComponentNames(const ComponentFactory& self, const std::string& className)
{
    std::set<std::string> allClassNames;
    std::transform(self.getRegistry().begin(), self.getRegistry().end(), std::inserter(allClassNames, allClassNames.begin()),
        [](const ComponentRegistrationData::SPtr& component){ return component->componentName; });
    const auto result = sofa::helper::getClosestMatch(className, std::vector(allClassNames.begin(), allClassNames.end()), 5, 0.6);
    std::vector<std::string> similarComponentNames;
    std::transform(result.begin(), result.end(), std::back_inserter(similarComponentNames),
        [](const auto& match) { return std::get<0>(match); } );
    return similarComponentNames;
}

bool knownIssues(const ComponentFactory& self, const std::string& clasName)
{
    auto uncreatableComponent = helper::lifecycle::uncreatableComponents.find(clasName);
    auto dealiasedComponent = helper::lifecycle::dealiasedComponents.find(clasName);

    const bool isUncreatable =  uncreatableComponent != helper::lifecycle::uncreatableComponents.end();
    const bool isDealiased = dealiasedComponent != helper::lifecycle::dealiasedComponents.end();

    if (isUncreatable)
    {
        msg_error(&self) << uncreatableComponent->second.getMessage();
        return true;
    }
    if (isDealiased)
    {
        msg_error(&self) << dealiasedComponent->second.getMessage();
        return true;
    }

    return false;
}

std::vector<ComponentRegistrationData::SPtr> selectCandidatesDeductionRules(
    const std::vector<ComponentRegistrationData::SPtr>& candidates,
    objectmodel::BaseContext* context,
    objectmodel::BaseObjectDescription* arg)
{
    std::vector<ComponentRegistrationData::SPtr> matchingCandidates;

    for (const auto& candidate : candidates)
    {
        // Check if the component has a rule AND if that rule applies.
        if (const auto rule = candidate->templateDeductionRule;
            rule && rule->doesComponentComplyWith(context, arg))
        {
            matchingCandidates.push_back(candidate);
        }
    }

    return matchingCandidates;
}

/**
 * Apply a filter on a list of potential candidates. If the filtered list has a unique element, a
 * component will be created based on this element. Otherwise, the filtered list is further filtered
 * based on deduction rules.
 */
ComponentRegistrationData::SPtr applyFilter(
    const ComponentFactory& self,
    const std::string& componentName,
    const std::vector<ComponentRegistrationData::SPtr>& candidates,
    objectmodel::BaseContext* context,
    objectmodel::BaseObjectDescription* arg,
    const ComponentFilter& filter)
{
    const auto filteredCandidates = filter.filter(candidates, context, arg);

    if (!filteredCandidates.empty())
    {
        if (filteredCandidates.size() == 1)
        {
            // No ambiguity: The unique candidate is returned
            return filteredCandidates.front();
        }
        else
        {
            // Multiple candidates: Use deduction rules to resolve ambiguity.
            auto deducedCandidates = selectCandidatesDeductionRules(filteredCandidates, context, arg);
            if (!deducedCandidates.empty())
            {
                msg_warning_when(deducedCandidates.size() > 1, &self)
                    << "Attempt to create component '" << componentName
                    << "', however multiple potential candidates match the provided attributes ("
                    << sofa::helper::join(
                        deducedCandidates.begin(), deducedCandidates.end(), ", ")
                    << "). The first one is selected.";
                return deducedCandidates.front();
            }
            else
            {
                msg_warning_when(filteredCandidates.size() > 1, &self)
                    << "Attempt to create component '" << componentName
                    << "', however multiple potential candidates match the provided attributes ("
                    << sofa::helper::join(
                        filteredCandidates.begin(), filteredCandidates.end(),
                        [](const ComponentRegistrationData::SPtr& desc){ std::stringstream ss; ss << *desc; return ss.str(); }, ", ")
                    << "). The first one is selected.";
                // None of the deduction rules matches: returning the first filtered candidate
                return filteredCandidates.front();
            }
        }
    }
    return nullptr;
}

void reportMissingComponents(ComponentFactory& self, const std::string& componentName)
{
    std::stringstream ss;
    ss << "Cannot create component '" << componentName << "': '" << componentName << "' not found in the factory registry.";

    const auto similarNames = similarComponentNames(self, componentName);
    if (!similarNames.empty())
    {
        ss << " Suggestion: Components were found with similar names: " << sofa::helper::join(similarNames, ", ");
    }

    msg_error(&self) << ss.str();
}

bool filterUnloadedPluginsCandidates(const ComponentFactory& self,
    const std::string& componentName,
    const std::vector<ComponentRegistrationData::SPtr>& candidates)
{
    auto candidatesWithoutUnloadedPlugins = candidates;
    std::erase_if(candidatesWithoutUnloadedPlugins, [](const ComponentRegistrationData::SPtr& candidate)
    {
        return helper::system::PluginManager::getInstance().isPluginUnloaded(candidate->componentModule);
    });

    if (candidatesWithoutUnloadedPlugins.empty())
    {
        std::set<std::string> unloadedPlugins;
        std::transform(candidates.begin(), candidates.end(), std::inserter(unloadedPlugins, unloadedPlugins.begin()),
            [](const ComponentRegistrationData::SPtr& component) { return component->componentModule; });
        const auto unloadedPluginsString = sofa::helper::join(unloadedPlugins.begin(), unloadedPlugins.end(), ", ");
        msg_error(&self) << "Attempted to create component '" << componentName
            << "' but all potential candidates rely on component from currently unloaded plugins:" << unloadedPluginsString << "]";
        return false;
    }
    return true;
}

}


ComponentRegistrationData::SPtr ComponentFactory::findComponent(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    if (!arg) return nullptr;

    std::string componentName, moduleName;
    {
        const char* typeAttribute = arg->getAttribute( "type", nullptr);
        if (typeAttribute == nullptr) return nullptr;
        extractModuleName(std::string{typeAttribute}, componentName, moduleName);
    }

    if (knownIssues(*this, componentName)) return nullptr;

    // 1. Ensure plugins are loaded
    if (auto it = helper::lifecycle::movedComponents.find(componentName);
        it != helper::lifecycle::movedComponents.end())
    {
        // autoLoadPlugin(*this, );
    }

    if (!moduleName.empty())
    {
        autoLoadPlugin(*this, moduleName);
    }

    // 2. Get initial candidates
    std::vector<ComponentRegistrationData::SPtr> candidates = getComponentsFromName(*this, componentName, moduleName);
    if (candidates.empty())
    {
        reportMissingComponents(*this, componentName);
        return nullptr;
    }

    // 3. Filter out candidates from unloaded plugins
    if (!filterUnloadedPluginsCandidates(*this, componentName, candidates))
    {
        return nullptr;
    }

    // 4. Sort by priority
    // In case of ambiguity (multiple candidates), sorting candidates will allow returning the
    // component with the highest priority.
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a->instantiationPriority > b->instantiationPriority; });

    // 5. Apply template matching filters
    static std::array<std::unique_ptr<ComponentFilter>, 4> filters {
        std::make_unique<ExactTemplateMatchFilter>(), //Exact Template Match (Highest Priority).
        std::make_unique<LegacyTemplateKeywordFilter>(), //Selection by Legacy 'template' Keyword (Medium-High Priority)
        std::make_unique<PartialTemplateMatchFilter>(), //Partial Template Matching (Medium Priority)
        std::make_unique<NoFilter>() //General Template Deduction (Lowest Priority)
    };

    for (const auto& filter : filters)
    {
        if (auto component = applyFilter(*this, componentName, candidates, context, arg, *filter))
        {
            return component;
        }
    }

    // Final fallback
    if (!candidates.empty())
    {
        return candidates.front();
    }

    // Final failure
    msg_error() << "Could not find or select a unique component for '" << componentName << "'";

    return nullptr;
}

objectmodel::BaseComponent::SPtr ComponentFactory::createComponent(
    objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    if (auto componentData = findComponent(context, arg))
    {
        auto component = componentData->creator->create();

        if (component)
        {
            if (context)
            {
                context->addObject(component);
            }

            msg_warning_when(componentData->componentModule.empty(), component.get()) << "Module name is empty";

            component->parse(arg);
            component->d_factoryName.setValue(componentData->componentName);
        }

        return component;
    }
    return nullptr;
}

objectmodel::BaseComponent::SPtr ComponentFactory::createObject(
    objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    return this->createComponent(context, arg);
}

bool ComponentFactory::hasCreator(const std::string& classname) const
{
    return std::any_of(m_registry.begin(), m_registry.end(),
        [&](const auto& component){ return component->componentName == classname; });
}

void ComponentFactory::getEntriesFromTarget(std::vector<ComponentRegistrationData::SPtr>& result,
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

std::string ComponentFactory::listClassesFromTarget(std::string target, std::string separator) const
{
    std::vector<ComponentRegistrationData::SPtr> entries;
    this->getEntriesFromTarget(entries, target);
    return sofa::helper::join(entries.begin(), entries.end(),
        [](const auto& entry) { std::stringstream ss; ss << *entry; return ss.str();}, separator);
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

ComponentFactory* MainComponentFactory::getInstance()
{
    static ComponentFactory instance;
    return &instance;
}

objectmodel::BaseComponent::SPtr MainComponentFactory::CreateComponent(
    objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    return getInstance()->createComponent(context, arg);
}

objectmodel::BaseComponent::SPtr MainComponentFactory::CreateObject(
    objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    return CreateComponent(context, arg);
}

bool MainComponentFactory::HasCreator(const std::string& classname)
{
    return getInstance()->hasCreator(classname);
}

}  // namespace sofa::core
