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
#include <set>
#include <string>
#include <optional>
#include <memory>
#include <vector>

namespace sofa::core
{

struct SOFA_CORE_API BaseTemplateDeductionRule
{
    virtual bool doesComponentComplyWith(
        objectmodel::BaseContext* context,
        objectmodel::BaseObjectDescription* arg) = 0;
};

struct SOFA_CORE_API ComponentRegistrationData
{
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

struct SOFA_CORE_API ComponentRegistrationDataBuilder : public ComponentRegistrationData
{
    ComponentRegistrationDataBuilder& setName(const std::string& _componentName)
    {
        this->componentName = _componentName;
        return *this;
    }

    ComponentRegistrationDataBuilder& addAlias(const std::string& alias)
    {
        this->aliases.insert(alias);
        return *this;
    }

    ComponentRegistrationDataBuilder& addTemplateAttribute(
        const std::string& templateAttribute, const std::string& value)
    {
        this->templateAttributes.emplace_back(templateAttribute, value);
        return *this;
    }

    template<class T>
    ComponentRegistrationDataBuilder& addTemplateAttribute(const std::string& templateAttribute)
    {
        return addTemplateAttribute(templateAttribute, T::Name());
    }

    ComponentRegistrationDataBuilder& setModuleName(const std::string& _moduleName)
    {
        this->componentModule = _moduleName;
        return *this;
    }

    ComponentRegistrationDataBuilder& setDescription(const std::string& _description)
    {
        this->description = _description;
        return *this;
    }

    ComponentRegistrationDataBuilder& addAuthor(const std::string& _author)
    {
        this->authors.insert(_author);
        return *this;
    }

    ComponentRegistrationDataBuilder& setLicense(const std::string& _license)
    {
        this->license = _license;
        return *this;
    }

    ComponentRegistrationDataBuilder& setDocumentationURL(const std::string& _documentationURL)
    {
        this->documentationURL.insert(_documentationURL);
        return *this;
    }
};

//to deprecate
struct SOFA_CORE_API LegacyComponentRegistrationData
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
    template<class T> LegacyComponentRegistrationData& add(bool = false)
    {
        m_componentCreators.push_back(new ComponentCreator<T>);
        m_componentNames.push_back(T::GetClass()->className);
        //to do: deal with templates
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
