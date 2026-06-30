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

    std::string componentNamespace;
    std::string componentModule;

    unsigned int instantiationPriority {};

    std::string description;
    std::set<std::string> authors;
    std::string license;
    std::set<std::string> documentationURL;

    std::shared_ptr<BaseTemplateDeductionRule> templateDeductionRule;

    std::unique_ptr<BaseComponentCreator> creator;

private:
    // ComponentRegistrationData() = default;
};

struct SOFA_CORE_API ComponentRegistrationDataBuilder
{
    ComponentRegistrationData::SPtr data;

    ComponentRegistrationDataBuilder(const std::string& componentName, const std::string& moduleName, const std::string& description, std::unique_ptr<BaseComponentCreator> creator)
        : data(std::make_shared<ComponentRegistrationData>())
    {
        data->componentName = componentName;
        data->componentModule = moduleName;
        data->description = description;
        data->creator = std::move(creator);
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

    template<class T>
    ComponentRegistrationDataBuilder& withDeductionRule()
    {
        data->templateDeductionRule = std::make_shared<T>();
        return *this;
    }

    ComponentRegistrationDataBuilder& withInstantiationPriority(unsigned int instantiationPriority)
    {
        data->instantiationPriority = instantiationPriority;
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

struct SOFA_CORE_API ComponentRegistrationDataModule
{
    ComponentRegistrationDataBuilder withDescription(const std::string& description)
    {
        return {m_componentName, m_moduleName, description, std::move(m_creator)};
    }
    ComponentRegistrationDataModule(const std::string& componentName, const std::string& moduleName, std::unique_ptr<BaseComponentCreator> creator)
        : m_componentName(componentName), m_moduleName(moduleName), m_creator(std::move(creator)) {}
private:
    std::string m_componentName;
    std::string m_moduleName;
    std::unique_ptr<BaseComponentCreator> m_creator;
};

struct SOFA_CORE_API ComponentRegistrationDataName
{
    ComponentRegistrationDataModule withModule(const std::string& moduleName)
    {
        return {m_componentName, moduleName, std::move(m_creator)};
    }
    ComponentRegistrationDataName(const std::string& componentName, std::unique_ptr<BaseComponentCreator> creator)
        : m_componentName(componentName)
        , m_creator(std::move(creator))
    {}
private:
    std::string m_componentName;
    std::unique_ptr<BaseComponentCreator> m_creator;
};

template<class Component>
ComponentRegistrationDataName CreateComponent(const std::string& componentName)
{
    std::unique_ptr<BaseComponentCreator> creator = std::make_unique<ComponentCreator<Component>>();
    return ComponentRegistrationDataName(componentName, std::move(creator));
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
