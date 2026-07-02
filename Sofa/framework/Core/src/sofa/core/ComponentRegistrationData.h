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

#include <sofa/core/config.h>
#include <sofa/core/ComponentCreator.h>
#include <sofa/core/TemplateDeductionRules.h>
#include <set>
#include <string>
#include <optional>
#include <memory>
#include <vector>

namespace sofa::core
{

struct SOFA_CORE_API ComponentRegistrationData
{
    friend class ComponentFactory;
    using SPtr = std::shared_ptr<ComponentRegistrationData>;

    std::string componentName;
    std::set<std::string> aliases;

    std::vector<std::pair<std::string, std::string>> templateAttributes;

    std::string componentModule;

    unsigned int instantiationPriority {};

    std::string description;
    std::set<std::string> authors;
    std::string license;
    std::set<std::string> documentationURL;

    std::shared_ptr<BaseTemplateDeductionRule> templateDeductionRule;

    std::unique_ptr<BaseComponentCreator> creator;
    const BaseClass* classData { nullptr };

private:
    // ComponentRegistrationData() = default;
};

struct SOFA_CORE_API ComponentRegistrationDataBuilder
{
    ComponentRegistrationData::SPtr data;

    ComponentRegistrationDataBuilder& withName(const std::string& name)
    {
        data->componentName = name;
        return *this;
    }

    ComponentRegistrationDataBuilder& withModule(const std::string& componentModule)
    {
        data->componentModule = componentModule;
        return *this;
    }

    ComponentRegistrationDataBuilder& withDescription(const std::string& description)
    {
        data->description = description;
        return *this;
    }

    ComponentRegistrationDataBuilder& addAlias(const std::string& alias)
    {
        data->aliases.insert(alias);
        return *this;
    }

    ComponentRegistrationDataBuilder& addTemplateAttribute(
        const std::string& templateAttribute, const std::string& value)
    {
        data->templateAttributes.emplace_back(templateAttribute, value);
        return *this;
    }

    template<class T>
    ComponentRegistrationDataBuilder& addTemplateAttribute(const std::string& templateAttribute)
    {
        return addTemplateAttribute(templateAttribute, T::Name());
    }

    ComponentRegistrationDataBuilder& addAuthor(const std::string& _author)
    {
        data->authors.insert(_author);
        return *this;
    }

    ComponentRegistrationDataBuilder& withLicense(const std::string& _license)
    {
        data->license = _license;
        return *this;
    }

    ComponentRegistrationDataBuilder& withDocumentationURL(const std::string& _documentationURL)
    {
        data->documentationURL.insert(_documentationURL);
        return *this;
    }

    ComponentRegistrationDataBuilder& withDeductionRule(const std::shared_ptr<BaseTemplateDeductionRule>& rule)
    {
        data->templateDeductionRule = rule;
        return *this;
    }

    ComponentRegistrationDataBuilder& withInstantiationPriority(unsigned int instantiationPriority)
    {
        data->instantiationPriority = instantiationPriority;
        return *this;
    }

    ComponentRegistrationDataBuilder& withClass(const BaseClass* classData)
    {
        data->classData = classData;
        return *this;
    }

    ComponentRegistrationDataBuilder& withCreator(std::unique_ptr<BaseComponentCreator> creator)
    {
        data->creator = std::move(creator);
        return *this;
    }

    operator ComponentRegistrationData::SPtr() const
    {
        return data;
    }
};

inline std::ostream& operator<<(std::ostream& os, const ComponentRegistrationData& data)
{
    os << data.componentName;
    if (!data.templateAttributes.empty())
    {
        os << "[";
        os << sofa::helper::join(data.templateAttributes.begin(), data.templateAttributes.end(),
            [](const std::pair<std::string, std::string>& pair)
            {
                return pair.first + "=" + pair.second;
            }, ',');
        os << "]";
    }
    return os;
}

template <class T>
concept HasTemplateDeductionRule = requires {
    typename T::TemplateDeductionRule;
};

template<class Component>
ComponentRegistrationDataBuilder CreateComponent(const std::string& componentName)
{
    std::unique_ptr<BaseComponentCreator> creator = std::make_unique<ComponentCreator<Component>>();
    BaseClass* classData = Component::GetClass();

    std::shared_ptr<BaseTemplateDeductionRule> templateDeductionRule { nullptr };
    if constexpr (HasTemplateDeductionRule<Component>)
    {
        templateDeductionRule = std::make_shared<typename Component::TemplateDeductionRule>();
    }
    else
    {
        templateDeductionRule = std::make_shared<CanCreateDeductionRule<Component>>();
    }

    return ComponentRegistrationDataBuilder()
        .withName(componentName)
        .withClass(classData)
        .withDeductionRule(templateDeductionRule)
        .withCreator(std::move(creator));
}

template<class Component>
ComponentRegistrationDataBuilder CreateComponent()
{
    BaseClass* classData = Component::GetClass();
    return CreateComponent<Component>(classData->className);
}



/**************************************************************************************************/


//to deprecate
struct SOFA_CORE_DEPRECATED_OBJECTFACTORY_LEGACYREGISTRATIONDATA() SOFA_CORE_API LegacyComponentRegistrationData
{

    std::string m_description;
    explicit LegacyComponentRegistrationData(const std::string& description)
        : m_description(description)
    {}

    std::vector<BaseComponentCreator*> m_componentCreators;
    ~LegacyComponentRegistrationData()
    {
        for (auto* componentCreator : m_componentCreators)
        {
            delete componentCreator;
        }
    }

    std::vector<std::string> m_componentNames;
    std::vector<std::string> m_componentTemplates;
    std::vector<std::string> m_moduleNames;
    std::vector<unsigned int> m_instantiationPriority;
    std::vector<std::shared_ptr<BaseTemplateDeductionRule> > m_templateDeductionRules;
    template<class T> LegacyComponentRegistrationData& add(bool defaultTemplate = false)
    {
        m_componentCreators.push_back(new ComponentCreator<T>);
        m_componentNames.push_back(sofa::core::objectmodel::BaseClassNameHelper::getClassName<T>());
        m_componentTemplates.push_back(sofa::core::objectmodel::BaseClassNameHelper::getTemplateName<T>());
#ifdef SOFA_TARGET
        m_moduleNames.push_back(sofa_tostring(SOFA_TARGET));
#else
        m_moduleNames.emplace_back();
#endif
        m_templateDeductionRules.push_back(std::make_shared<CanCreateDeductionRule<T>>());
        m_instantiationPriority.push_back(std::numeric_limits<unsigned int>::max() * defaultTemplate);
        return *this;
    }

    std::string m_documentationURL;
    LegacyComponentRegistrationData& addDocumentationURL(const std::string& documentationURL)
    {
        m_documentationURL = documentationURL;
        return *this;
    }

    LegacyComponentRegistrationData& addDescription(const std::string& description)
    {
        m_description = description;
        return *this;
    }

    std::set<std::string> m_aliases;
    LegacyComponentRegistrationData& addAlias(const std::string& alias)
    {
        m_aliases.insert(alias);
        return *this;
    }

    std::string m_authors;
    LegacyComponentRegistrationData& addAuthor(const std::string& author)
    {
        m_authors = author;
        return *this;
    }

    std::string m_license;
    LegacyComponentRegistrationData& addLicense(const std::string& license)
    {
        m_license = license;
        return *this;
    }
};



}
