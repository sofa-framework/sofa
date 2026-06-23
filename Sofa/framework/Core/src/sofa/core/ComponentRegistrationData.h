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
    virtual bool doesTemplateDeductionApply(
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
    explicit LegacyComponentRegistrationData(const std::string&) {}

    template<class T> LegacyComponentRegistrationData& add(bool = false)
    {
        return *this;
    }

    LegacyComponentRegistrationData& addDocumentationURL(const std::string&)
    {
        return *this;
    }

    LegacyComponentRegistrationData& addDescription(const std::string&)
    {
        return *this;
    }

    LegacyComponentRegistrationData& addAlias(const std::string&)
    {
        return *this;
    }
};



}
