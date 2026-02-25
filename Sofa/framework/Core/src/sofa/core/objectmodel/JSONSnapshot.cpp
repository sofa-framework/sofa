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
#include <sofa/core/objectmodel/JSONSnapshot.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <string>
#include <stdexcept>
#include <iostream>
#include <sofa/helper/system/SetDirectory.h>

#include <sofa/core/objectmodel/Data.h>


namespace sofa::core::objectmodel
{


JSONSnapshot::JSONSnapshot()
{}
JSONSnapshot::~JSONSnapshot() = default;


void to_json(nlohmann::json& j, const BaseSnapshot::DataInfo& di )
{
    j.clear();
    j["name"]  = di.name;
    j["type"]  = di.type;
    j["value"] = di.value;
}

void to_json(nlohmann::json& j, const BaseSnapshot::LinkInfo& li )
{
    j.clear();
    j["name"]       = li.name;
    j["type"] = li.type;
    j["value"]       = li.value;
}

void to_json(nlohmann::json& j, const BaseSnapshot::SnapshotObject& so )
{
    j.clear();
    j["name"] = so.m_name;
    j["datas"] = so.m_dataContainer;
    j["links"] = so.m_linkContainer;
}

void to_json(nlohmann::json& j, const BaseSnapshot::SnapshotNode& sn)
{
    j.clear();
    j["name"] = sn.m_name;
    j["datas"] = sn.m_dataContainer;
    j["links"] = sn.m_linkContainer;
    j["components"] = sn.components;

    j["children"] = nlohmann::json::array();
    for (const auto& childPtr : sn.children)
    {
        if(childPtr)
        {
            j["children"].push_back(*childPtr);
        }
        else
        {
            j["children"].push_back(nullptr);
        }
    }
}

void to_json(nlohmann::json& j, const std::shared_ptr<BaseSnapshot::SnapshotNode>& sn)
{
    j.clear();
    j["name"] = sn->m_name;
    j["datas"] = sn->m_dataContainer;
    j["links"] = sn->m_linkContainer;
    j["components"] = sn->components;
    j["children"] = nlohmann::json::array();
    for (const auto& childPtr : sn->children)
    {
        if(childPtr)
        {
            j["children"].push_back(*childPtr);
        }
        else
        {
            j["children"].push_back(nullptr);
        }
    }
}

void JSONSnapshot::exportTo(const std::string filename)
{
    nlohmann::json j = *m_graphRoot ;

    std::ofstream file(filename);
    file << j.dump(5);
    file.close();
}

void JSONSnapshot::importSnapshot(const std::string filename)
{
    std::cout << "importSnapshot" << std::endl;    
}


void from_json(const nlohmann::json& j, BaseSnapshot::DataInfo& di)
{
    di.name = j.value("name", "");
    di.type = j.value("type", "");
    di.value = j.value("value", "");
}

void from_json(const nlohmann::json& j, BaseSnapshot::LinkInfo& li)
{
    li.name = j.value("name", "");
    li.type = j.value("type", "");
    li.value = j.value("value", "");
}

void from_json(const nlohmann::json& j, BaseSnapshot::SnapshotObject& so)
{
    so.m_name = j.value("name", "");
    
    if (j.contains("datas") && j["datas"].is_array())
    {
        so.m_dataContainer.clear();
        for (const auto& dataJson : j["datas"])
        {
            BaseSnapshot::DataInfo di;
            from_json(dataJson, di);
            so.m_dataContainer.push_back(di);
        }
    }
    
    if (j.contains("links") && j["links"].is_array())
    {
        so.m_linkContainer.clear();
        for (const auto& linkJson : j["links"])
        {
            BaseSnapshot::LinkInfo li;
            from_json(linkJson, li);
            so.m_linkContainer.push_back(li);
        }
    }
}

void from_json(const nlohmann::json& j, BaseSnapshot::SnapshotNode& sn)
{
    sn.m_name = j.value("name", "");
    
    // Parser les données
    if (j.contains("datas") && j["datas"].is_array())
    {
        sn.m_dataContainer.clear();
        for (const auto& dataJson : j["datas"])
        {
            BaseSnapshot::DataInfo di;
            from_json(dataJson, di);
            sn.m_dataContainer.push_back(di);
        }
    }
    
    // Parser les liens
    if (j.contains("links") && j["links"].is_array())
    {
        sn.m_linkContainer.clear();
        for (const auto& linkJson : j["links"])
        {
            BaseSnapshot::LinkInfo li;
            from_json(linkJson, li);
            sn.m_linkContainer.push_back(li);
        }
    }
    
    // Parser les composants
    if (j.contains("components") && j["components"].is_array())
    {
        sn.components.clear();
        for (const auto& compJson : j["components"])
        {
            BaseSnapshot::SnapshotObject so;
            from_json(compJson, so);
            sn.components.push_back(so);
        }
    }
    
    // Parser les enfants (récursivement)
    sn.children.clear();
    if (j.contains("children") && j["children"].is_array())
    {
        for (const auto& childJson : j["children"])
        {
            if (!childJson.is_null())
            {
                auto child = std::make_shared<BaseSnapshot::SnapshotNode>();
                from_json(childJson, *child);
                sn.children.push_back(child);
            }
        }
    }
}

void JSONSnapshot::importFrom(const std::string filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open file " << filename << " for reading\n";
        return;
    }

    nlohmann::json jsonRoot;
    file >> jsonRoot;
    file.close();

    if (!m_graphRoot)
    {
        m_graphRoot = std::make_shared<BaseSnapshot::SnapshotNode>();
    }


    if (jsonRoot.is_object() && !jsonRoot.empty())
    {
        from_json(jsonRoot, *m_graphRoot);
    }
    else
    {
        std::cerr << "ERROR: Invalid JSON format in " << filename << "\n";
        return;
    }

    std::cout << "JSON imported successfully from: " << filename << std::endl;

    
}




} // namespace sofa::core::objectmodel