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
#include <sofa/core/ObjectFactory.h>
#include <json.h>

namespace sofa::core
{

namespace objectmodel
{

inline void to_json(nlohmann::json& json,
                    const objectmodel::BaseClass& baseClass)
{
    json["namespaceName"] = baseClass.namespaceName;
    json["typeName"] = baseClass.typeName;
    json["className"] = baseClass.className;
    json["templateName"] = baseClass.templateName;
    json["shortName"] = baseClass.shortName;
}

inline void to_json(nlohmann::json& json,
                    const objectmodel::BaseData* data)
{
    json["name"] = data->m_name;
    json["group"] = data->group;
    json["help"] = data->help;
    json["type"] = data->getValueTypeString();
    json["defaultValue"] = data->getDefaultValueString();
}

inline void to_json(nlohmann::json& json,
                    const objectmodel::BaseLink* link)
{
    json["name"] = link->getName();
    json["help"] = link->getHelp();
}

inline void to_json(nlohmann::json& json,
                    const objectmodel::BaseObject::SPtr& object)
{
    json["data"] = object->getDataFields();
    json["link"] = object->getLinks();
}

}

inline void to_json(nlohmann::json& json,
                    const sofa::core::ObjectFactory::Creator::SPtr& creator)
{
    json["target"] = creator->getTarget();
    json["class"] = *creator->getClass();

    sofa::core::objectmodel::BaseObjectDescription desc;
    if (const auto object = creator->createInstance(nullptr, &desc))
    {
        json["object"] = object;
    }
}

inline void to_json(nlohmann::json& json,
                    const sofa::core::ObjectFactory::ClassEntry::SPtr& entry)
{
    json["className"] = entry->className;
    json["description"] = entry->description;

    json["creator"] = entry->creatorMap;
}

std::string ObjectFactoryJson::dump(ObjectFactory* factory)
{
    std::vector<sofa::core::ObjectFactory::ClassEntry::SPtr> entries;
    factory->getAllEntries(entries);

    const nlohmann::json json = entries;

    return json.dump();
}
}
