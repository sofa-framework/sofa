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

class SOFA_CORE_API ComponentRegistrationData
{
public:

    std::string componentName;
    std::set<std::string> aliases;
    std::string description;

    std::set<std::string> authors;
    std::string license;
    std::set<std::string> documentationURL;
    std::string componentNamespace;
    std::string componentModule;

    std::optional<std::size_t> defaultTemplateId;

    std::vector<std::unique_ptr<BaseComponentCreator>> creators;

    /// Start the registration by giving the description of this class.
    explicit ComponentRegistrationData(const std::string& description)
    {
        if (!description.empty())
        {
            addDescription(description);
        }
    }

    ComponentRegistrationData(const ComponentRegistrationData&) = delete;
    void operator=(const ComponentRegistrationData&) = delete;

    /// Add an alias name for this class
    ComponentRegistrationData& addAlias(std::string val)
    {
        aliases.insert(val);
        return *this;
    }

    /// Add more descriptive text about this class
    ComponentRegistrationData& addDescription(const std::string& val)
    {
        if (description.empty())
        {
            description = val;
        }
        else
        {
            dmsg_error("ComponentRegistrationData") << "Trying to add multiple descriptions for a single component whereas only one is supported";
        }
        return *this;
    }

    /// Specify a list of authors (separated with spaces)
    ComponentRegistrationData& addAuthor(std::string val)
    {
        authors.insert(val);
        return *this;
    }

    /// Specify a license (LGPL, GPL, ...)
    ComponentRegistrationData& addLicense(std::string val)
    {
        if (license.empty())
        {
            license = val;
        }
        else
        {
            dmsg_error("ComponentRegistrationData") << "Trying to add multiple licenses for a single component whereas only one is supported";
        }
        return *this;
    }

    /// Specify a documentation URL
    ComponentRegistrationData& addDocumentationURL(std::string url)
    {
        documentationURL.insert(url);
        return *this;
    }

    /// Add a template instantiation of this class.
    ///
    /// \param defaultTemplate    set to true if this should be the default instance when no template name is given.
    template<class RealObject>
    ComponentRegistrationData& add(bool defaultTemplate = false)
    {
#ifdef SOFA_TARGET
        const std::string target = sofa_tostring(SOFA_TARGET);

        if (!target.empty())
        {
            componentNamespace = target;
            componentModule = target;
        }
#else
        dmsg_warning("ComponentFactory") << "Module name cannot be found when registering "
                << RealObject::GetClass()->className << "<" << RealObject::GetClass()->templateName << "> into the component factory";
#endif

        const std::string classname = sofa::core::objectmodel::BaseClassNameHelper::getClassName<RealObject>();
        if (componentName.empty())
        {
            componentName = classname;
        }
        else
        {
            if (componentName != classname)
            {
                msg_error("ComponentFactory") << "Trying to define a class (" << classname << ") unrelated to " << componentName;
                return *this;
            }
        }

        creators.push_back(std::unique_ptr<BaseComponentCreator>(new ComponentCreator<RealObject>));

        if (defaultTemplate)
        {
            if (defaultTemplateId.has_value())
            {
                msg_error("ComponentFactory") << "Trying to define a default template for "
                    << RealObject::GetClass()->className << " whereas one was already defined before";
            }
            else
            {
                defaultTemplateId = creators.size() - 1;
            }
        }

        return *this;
    }

};

}
