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
#include <sofa/core/ObjectFactoryJson.h>
#include <sofa/core/ComponentFactory.h>
#include <nlohmann/json.hpp>
#include <sofa/core/CategoryLibrary.h>


namespace sofa::core
{

namespace objectmodel
{

inline void to_json(nlohmann::json& json,
                    const objectmodel::BaseData* data)
{
    if (data)
    {
        json["name"] = data->m_name;
        json["group"] = data->group;
        json["help"] = data->help;
        json["type"] = data->getValueTypeString();
        json["defaultValue"] = data->getDefaultValueString();
    }
}

inline void to_json(nlohmann::json& json,
                    const objectmodel::BaseLink* link)
{
    if (link)
    {
        json["name"] = link->getName();
        json["help"] = link->getHelp();
        json["destinationTypeName"] = link->getValueTypeString();
    }
}

inline void to_json(nlohmann::json& json,
                    const objectmodel::BaseComponent::SPtr& object)
{
    if (object)
    {
        json["data"] = object->getDataFields();
        json["link"] = object->getLinks();
    }
}

}

inline void to_json(nlohmann::json& json,
                    const sofa::core::ComponentRegistrationData::SPtr& entry)
{
    if (entry)
    {
        json["name"] = entry->componentName;
        json["attributes"] = entry->templateAttributes;
        json["module"] = entry->componentModule;
        json["description"] = entry->description;
        json["documentationURL"] = entry->documentationURL;
        // json["templateDeductionRule"] = entry->templateDeductionRule;

        if (entry->creator)
        {
            auto component = entry->creator->create();
            json["component"] = component;
        }
    }
}

std::string ObjectFactoryJson::dump(ComponentFactory* factory)
{
    if (!factory)
    {
        msg_error("ObjectFactoryJson") << "Invalid factory: cannot dump to json";
        return {};
    }

    const auto& registry = factory->getRegistry();

    const nlohmann::json json = registry;

    std::string dump{};

    try
    {
        dump = json.dump();
    }
    catch (const nlohmann::json::type_error& e)
    {
        msg_error("ObjectFactoryJson") << "Error while dumping json from the object factory: " << e.what();
        dump = json.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    }

    return dump;
}
}
