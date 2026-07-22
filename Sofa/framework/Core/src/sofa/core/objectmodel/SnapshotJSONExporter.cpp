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
#include <sofa/core/objectmodel/SnapshotJSONExporter.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::MessageDispatcher ;
using sofa::helper::logging::Message ;

#define ERROR_LOG_SIZE 100

namespace sofa::core::objectmodel
{



void to_json(nlohmann::json& j, const Snapshot::DataInfo& di )
{
    j.clear();
    j["name"]  = di.name;
    j["type"]  = di.type;
    j["value"] = di.value;
}

void to_json(nlohmann::json& j, const Snapshot::LinkInfo& li )
{
    j.clear();
    j["name"]       = li.name;
    j["type"] = li.type;
    j["value"]       = li.value;
}

void to_json(nlohmann::json& j, const Snapshot::SnapshotObject& so )
{
    j.clear();
    j["name"] = so.m_name;
    j["datas"] = so.m_dataContainer;
    j["links"] = so.m_linkContainer;
    j["slaves"] = nlohmann::json::array();
    for (const auto& childPtr : so.m_components)
    {
        if(childPtr)
        {
            j["slaves"].push_back(*childPtr);
        }
        else
        {
            j["slaves"].push_back(nullptr);
        }
    }

}

void to_json(nlohmann::json& j, const Snapshot::SnapshotNode& sn)
{
    j.clear();
    j["name"] = sn.m_name;
    j["datas"] = sn.m_dataContainer;
    j["links"] = sn.m_linkContainer;

    j["components"] = nlohmann::json::array();
    for (const auto& childPtr : sn.m_components)
    {
        if(childPtr)
        {
            j["components"].push_back(*childPtr);
        }
        else
        {
            j["components"].push_back(nullptr);
        }
    }

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

 // Maybe this is not necessary : do for nlohmann::json j = *snapshot.m_graphRoot ;
void to_json(nlohmann::json& j, const std::shared_ptr<Snapshot::SnapshotNode>& sn)
{
    j.clear();
    j["name"] = sn->m_name;
    j["datas"] = sn->m_dataContainer;
    j["links"] = sn->m_linkContainer;

    j["components"] = nlohmann::json::array();
    for (const auto& childPtr : sn->m_components)
    {
        if(childPtr)
        {
            j["components"].push_back(*childPtr);
        }
        else
        {
            j["components"].push_back(nullptr);
        }
    }

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

void exportToJSON(const Snapshot& snapshot, const std::string& filename)
{
    std::cout << snapshot.m_graphRoot->m_name << std::endl;
    std::cout << snapshot.m_graphRoot->children.size() << std::endl;

    for (const auto& c : snapshot.m_graphRoot->children)
    {
        std::cout << "  child = " << c->m_name << std::endl;
    }
    nlohmann::json j = snapshot.m_graphRoot ;

    std::ofstream file(filename);
    file << j.dump(5);
    file.close();
}

void from_json(const nlohmann::json& j, Snapshot::DataInfo& di)
{
    di.name = j.value("name", "");
    di.type = j.value("type", "");
    di.value = j.value("value", "");
}

void from_json(const nlohmann::json& j, Snapshot::LinkInfo& li)
{
    li.name = j.value("name", "");
    li.type = j.value("type", "");
    li.value = j.value("value", "");
}

void from_json(const nlohmann::json& j, Snapshot::SnapshotObject& so)
{
    so.m_name = j.value("name", "");
    
    if (j.contains("datas") && j["datas"].is_array())
    {
        so.m_dataContainer.clear();
        for (const auto& dataJson : j["datas"])
        {
            Snapshot::DataInfo di;
            from_json(dataJson, di);
            so.m_dataContainer.push_back(di);
        }
    }
    
    if (j.contains("links") && j["links"].is_array())
    {
        so.m_linkContainer.clear();
        for (const auto& linkJson : j["links"])
        {
            Snapshot::LinkInfo li;
            from_json(linkJson, li);
            so.m_linkContainer.push_back(li);
        }
    }

    so.m_components.clear();
    if (j.contains("slaves") && j["slaves"].is_array())
    {
        for (const auto& childJson : j["slaves"])
        {
            if (!childJson.is_null())
            {
                auto child = std::make_shared<Snapshot::SnapshotNode>();
                from_json(childJson, *child);
                so.m_components.push_back(child);
            }
        }
    }
}

void from_json(const nlohmann::json& j, Snapshot::SnapshotNode& sn)
{
    sn.m_name = j.value("name", "");
    
    if (j.contains("datas") && j["datas"].is_array())
    {
        sn.m_dataContainer.clear();
        for (const auto& dataJson : j["datas"])
        {
            Snapshot::DataInfo di;
            from_json(dataJson, di);
            sn.m_dataContainer.push_back(di);
        }
    }
    
    if (j.contains("links") && j["links"].is_array())
    {
        sn.m_linkContainer.clear();
        for (const auto& linkJson : j["links"])
        {
            Snapshot::LinkInfo li;
            from_json(linkJson, li);
            sn.m_linkContainer.push_back(li);
        }
    }

    sn.m_components.clear();
    if (j.contains("components") && j["components"].is_array())
    {
        for (const auto& childJson : j["components"])
        {
            if (!childJson.is_null())
            {
                auto child = std::make_shared<Snapshot::SnapshotObject>();
                from_json(childJson, *child);
                sn.m_components.push_back(child);
            }
        }
    }
    
    sn.children.clear();
    if (j.contains("children") && j["children"].is_array())
    {
        for (const auto& childJson : j["children"])
        {
            if (!childJson.is_null())
            {
                auto child = std::make_shared<Snapshot::SnapshotNode>();
                from_json(childJson, *child);
                sn.children.push_back(child);
            }
        }
    }
}
    
void importFrom(Snapshot& snapshot, const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        msg_error("SnapshotJSONExporter") << "ERROR: Cannot open file " << filename << " for reading";
        return;
    }

    nlohmann::json jsonRoot;
    file >> jsonRoot;
    file.close();

    if (!snapshot.m_graphRoot)
    {
        snapshot.m_graphRoot = std::make_shared<Snapshot::SnapshotNode>();
    }

    if (jsonRoot.is_object() && !jsonRoot.empty())
    {
        from_json(jsonRoot, *snapshot.m_graphRoot);
    }
    else
    {
        msg_error("SnapshotJSONExporter") << "Invalid JSON format in " << filename ;
        return;
    }

    msg_info("SnapshotJSONExporter") << "JSON imported successfully from: " << filename;
}

std::string fileToString(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        msg_error("SnapshotJSONExporter") << "Cannot open file " << filename << " for reading";
        return "";
    }

    nlohmann::json jsonRoot;
    file >> jsonRoot;
    file.close();
    return to_string(jsonRoot);
}

std::string snapshotToString(const Snapshot& snapshot)
{
    nlohmann::json j = snapshot.m_graphRoot ;
    return to_string(j);
}

void exportToJSON(const std::map<std::string, std::shared_ptr<Snapshot>>& snapshots, const std::string& filename)
{
    std::ofstream file(filename);

    nlohmann::json j_all = nlohmann::json::array();

    for (const auto& snapshotJson : snapshots)
    {
        j_all.push_back(snapshotJson.second->m_graphRoot);
    }
    file << j_all.dump(5);
    file.close();
}

void importFrom(std::map<std::string, std::shared_ptr<Snapshot>>& snapshots, const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        msg_error("SnapshotJSONExporter") << "Cannot open file " << filename << " for reading";
        return;
    }

    nlohmann::json j_all = nlohmann::json::array();
    file >> j_all;
    file.close();

    snapshots.clear();

    int index = 0;

    for (const auto& snapshotJson : j_all)
    {
        auto snapshot = std::make_shared<Snapshot>();

        from_json(snapshotJson, *snapshot->m_graphRoot);

        std::string id;

        if (snapshot->m_graphRoot)
            id = snapshot->m_graphRoot->m_name;

        if (id.empty())
            id="snapshot_"+ std::to_string(index++);
        snapshot->m_graphRoot->m_name = id;
        snapshots[id] = snapshot;
    }
}

} // namespace sofa::core::objectmodel
