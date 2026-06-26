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
        ComponentDescription::SPtr component = std::make_shared<ComponentDescription>();

        component->componentName = ro.m_componentNames[i];
        component->aliases = ro.m_aliases;
        component->componentNamespace = "";
        component->componentModule = "";

        component->description = ro.m_description;
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

        component->creator = creator->clone();

        this->m_registry.push_back(component);
    }

    return true;

}

namespace
{


std::vector<ComponentDescription::SPtr> getComponentsFromName(
    const ComponentFactory& self,
    const std::string& componentName)
{
    std::vector<ComponentDescription::SPtr> result;

    std::string componentToSearch = componentName;

    using sofa::helper::lifecycle::renamedComponents;
    auto renamedComponent = renamedComponents.find(componentName);
    if( renamedComponent != renamedComponents.end() )
    {
        componentToSearch = renamedComponent->second.getNewName();
    }

    for (const auto& component : self.getRegistry())
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

std::vector<ComponentDescription::SPtr> selectCandidatesFromTemplateAttributes(const std::vector<ComponentDescription::SPtr>& candidates, objectmodel::BaseObjectDescription* arg)
{
    std::vector<ComponentDescription::SPtr> exactlyMatchingCandidates;

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
            exactlyMatchingCandidates.push_back(candidate);
        }

    }

    return exactlyMatchingCandidates;
}

std::vector<ComponentDescription::SPtr> selectCandidatesFromPartialTemplateAttributes(const std::vector<ComponentDescription::SPtr>& candidates, objectmodel::BaseObjectDescription* arg)
{
    std::vector<ComponentDescription::SPtr> partiallyMatchingCandidates;

    for (const auto& candidate : candidates)
    {
        for (const auto& [attribute, value] : candidate->templateAttributes)
        {
            const char* attr = arg->getAttribute(attribute, nullptr);
            if (attr != nullptr)
            {
                const std::string attrStr { attr };
                if (defaulttype::TemplateAliases::resolveAlias(attrStr) == value)
                {
                    partiallyMatchingCandidates.push_back(candidate);
                    break;
                }
            }
        }
    }

    return partiallyMatchingCandidates;
}

std::vector<ComponentDescription::SPtr> selectCandidatesTemplateKeyword(
    const std::vector<ComponentDescription::SPtr>& candidates,
    objectmodel::BaseObjectDescription* arg)
{
    const char* templateAttr = arg->getAttribute("template", nullptr);
    if (!templateAttr)
        return {};

    std::string templateAttrStr { templateAttr };
    templateAttrStr = defaulttype::TemplateAliases::resolveAlias(templateAttrStr);

    std::vector<ComponentDescription::SPtr> matchingCandidates;

    for (const auto& candidate : candidates)
    {
        const auto templateList = sofa::helper::join(
            candidate->templateAttributes.begin(), candidate->templateAttributes.end(),
            [](const auto& attr){ return attr.second; }, ',');
        if (templateAttrStr == templateList)
        {
            matchingCandidates.push_back(candidate);
        }
    }

    return matchingCandidates;
}

std::vector<ComponentDescription::SPtr> selectCandidatesDeductionRules(
    const std::vector<ComponentDescription::SPtr>& candidates,
    objectmodel::BaseContext* context,
    objectmodel::BaseObjectDescription* arg)
{
    std::vector<ComponentDescription::SPtr> matchingCandidates;

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

auto createComponentFrom(const ComponentDescription::SPtr& desc, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    auto component = desc->creator->create();

    if (component)
    {
        if (context)
        {
            context->addObject(component);
        }

        msg_warning_when(desc->componentModule.empty(), component.get()) << "Module name is empty";

        component->parse(arg);
    }

    return component;
}

std::vector<std::string> similarComponentNames(const ComponentFactory& self, const std::string& className)
{
    std::set<std::string> allClassNames;
    std::transform(self.getRegistry().begin(), self.getRegistry().end(), std::inserter(allClassNames, allClassNames.begin()),
        [](const ComponentDescription::SPtr& component){ return component->componentName; });
    const auto result = sofa::helper::getClosestMatch(className, std::vector(allClassNames.begin(), allClassNames.end()));
    std::vector<std::string> similarComponentNames;
    std::transform(result.begin(), result.end(), std::back_inserter(similarComponentNames),
        [](const auto& match) { return std::get<0>(match); } );
    return similarComponentNames;
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

    std::string inputClassName {typeAttribute};
    std::string classname, pluginName;
    extractModuleName(inputClassName, classname, pluginName);

    if (!pluginName.empty())
    {
        autoLoadPlugin(*this, pluginName);
    }

    std::vector<ComponentDescription::SPtr> candidates = getComponentsFromName(*this, classname);

    if (candidates.empty())
    {
        std::stringstream ss;
        ss << "Cannot create component '" << classname << "': '" << classname << "' not found in the factory.";

        const auto similarNames = similarComponentNames(*this, classname);
        if (!similarNames.empty())
        {
            ss << " Some components were font with similar names: " << sofa::helper::join(similarNames, ", ");
        }

        msg_error() << ss.str();
        return nullptr;
    }

    std::erase_if(candidates, [](const ComponentDescription::SPtr& candidate)
    {
        return helper::system::PluginManager::getInstance().isPluginUnloaded(candidate->componentModule);
    });

    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a->instantiationPriority > b->instantiationPriority; });

    //exact template: could it be a template deduction rule?
    {
        const auto matchingTemplates = selectCandidatesFromTemplateAttributes(candidates, arg);

        if (!matchingTemplates.empty())
        {
            msg_warning_when(matchingTemplates.size() > 1) << "Multiple candidates with the same templates: " <<
                sofa::helper::join(matchingTemplates.begin(), matchingTemplates.end(), ",");
            return createComponentFrom(matchingTemplates.front(), context, arg);
        }
    }

    // Selection based on the legacy 'template' keyword
    {
        const auto templateCandidates = selectCandidatesTemplateKeyword(candidates, arg);
        if (!templateCandidates.empty())
        {
            msg_warning_when(templateCandidates.size() > 1) << "Multiple candidates with the same templates: " <<
                sofa::helper::join(templateCandidates.begin(), templateCandidates.end(), ",");
            return createComponentFrom(templateCandidates.front(), context, arg);
        }
    }

    //partial template matching
    {
        const auto matchingTemplates = selectCandidatesFromPartialTemplateAttributes(candidates, arg);

        // only one candidate matches partially: this is the one
        if (matchingTemplates.size() == 1)
        {
            return createComponentFrom(matchingTemplates.front(), context, arg);
        }
        // otherwise, we discriminate them using deduction rules
        else if (!matchingTemplates.empty())
        {
            auto deducedCandidates = selectCandidatesDeductionRules(matchingTemplates, context, arg);
            if (!deducedCandidates.empty())
            {
                msg_warning_when(deducedCandidates.size() > 1) << "Multiple candidates with the same templates: " <<
                    sofa::helper::join(deducedCandidates.begin(), deducedCandidates.end(), ",");
                return createComponentFrom(deducedCandidates.front(), context, arg);
            }
        }
    }

    // Template deduction:
    // So far, none of the candidates match the template attributes.
    // We select one of them automatically based on deduction rules
    {
        auto deducedCandidates = selectCandidatesDeductionRules(candidates, context, arg);

        if (!deducedCandidates.empty())
        {
            msg_warning_when(deducedCandidates.size() > 1) << "Multiple candidates with the same templates: " <<
                sofa::helper::join(deducedCandidates.begin(), deducedCandidates.end(), ",");
            return createComponentFrom(deducedCandidates.front(), context, arg);
        }
    }

    msg_error() << "Cannot create component '" << classname << "'";

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
