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
#include <iostream>
#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::MessageDispatcher ;
using sofa::helper::logging::Message ;

#define ERROR_LOG_SIZE 100

namespace sofa::core::objectmodel
{



void to_json(nlohmann::ordered_json& jsonFile, const Snapshot::DataInfo& dataInfo )
{
    jsonFile.clear();
    jsonFile["name"] = dataInfo.m_name;
    jsonFile["type"] = dataInfo.m_type;
    jsonFile["value"] = dataInfo.m_value;
}

void to_json(nlohmann::ordered_json& jsonFile, const Snapshot::LinkInfo& linkInfo )
{
    jsonFile.clear();
    jsonFile["name"]= linkInfo.m_name;
    jsonFile["type"]= linkInfo.m_type;
    jsonFile["value"]= linkInfo.m_value;
}



void to_json(nlohmann::ordered_json& jsonFile, const Snapshot::SnapshotObject& snapshotObject )
{
    jsonFile.clear();
    jsonFile["name"] = snapshotObject.m_name;
    jsonFile["classname"] = snapshotObject.m_className;
    jsonFile["pathname"] = snapshotObject.m_pathName;
    jsonFile["data"] = snapshotObject.m_dataContainer;
    jsonFile["links"] = snapshotObject.m_linkContainer;
    jsonFile["slaves"] = nlohmann::json::array();
    for (const auto& childPtr : snapshotObject.m_objects)
    {
        if(childPtr)
        {
            jsonFile["slaves"].push_back(*childPtr);
        }
        else
        {
            jsonFile["slaves"].push_back(nullptr);
        }
    }

}


void to_json(nlohmann::ordered_json& jsonFile, const Snapshot::SnapshotNode& snapshotNode)
{
    jsonFile.clear();
    jsonFile["name"] = snapshotNode.m_name;
    jsonFile["classname"] = snapshotNode.m_className;
    jsonFile["pathname"] = snapshotNode.m_pathName;
    jsonFile["data"] = snapshotNode.m_dataContainer;
    jsonFile["links"] = snapshotNode.m_linkContainer;

    jsonFile["components"] = nlohmann::json::array();
    for (const auto& childPtr : snapshotNode.m_objects)
    {
        if (childPtr)
            jsonFile["components"].push_back(*childPtr);
        else
            jsonFile["components"].push_back(nullptr);
    }

    jsonFile["children"] = nlohmann::json::array();
    for (const auto& childPtr : snapshotNode.m_children)
    {
        if (childPtr)
            jsonFile["children"].push_back(*childPtr);
        else
            jsonFile["children"].push_back(nullptr);
    }
}

void from_json(const nlohmann::json& jsonFile, Snapshot::DataInfo& dataInfo)
{
    dataInfo.m_name = jsonFile.value("name", "");
    dataInfo.m_type = jsonFile.value("type", "");
    dataInfo.m_value = jsonFile.value("value", "");
}

void from_json(const nlohmann::json& jsonFile, Snapshot::LinkInfo& linkInfo)
{
    linkInfo.m_name = jsonFile.value("name", "");
    linkInfo.m_type = jsonFile.value("type", "");
    linkInfo.m_value = jsonFile.value("value", "");
}

void from_json(const nlohmann::json& jsonFile, Snapshot::SnapshotObject& snapshotObject)
{
    snapshotObject.m_name = jsonFile.value("name", "");
    snapshotObject.m_className = jsonFile.value("classname", "");
    snapshotObject.m_pathName = jsonFile.value("pathname","");

    if (jsonFile.contains("data") && jsonFile["data"].is_array())
    {
        snapshotObject.m_dataContainer.clear();
        for (const auto& dataJson : jsonFile["data"])
        {
            Snapshot::DataInfo dataInfo;
            from_json(dataJson, dataInfo);
            snapshotObject.m_dataContainer.push_back(dataInfo);
        }
    }

    if (jsonFile.contains("links") && jsonFile["links"].is_array())
    {
        snapshotObject.m_linkContainer.clear();
        for (const auto& linkJson : jsonFile["links"])
        {
            Snapshot::LinkInfo linkInfo;
            from_json(linkJson, linkInfo);
            snapshotObject.m_linkContainer.push_back(linkInfo);
        }
    }

    snapshotObject.m_objects.clear();
    if (jsonFile.contains("slaves") && jsonFile["slaves"].is_array())
    {
        for (const auto& childJson : jsonFile["slaves"])
        {
            if (!childJson.is_null())
            {
                auto child = std::make_shared<Snapshot::SnapshotNode>();
                from_json(childJson, *child);
                snapshotObject.m_objects.push_back(child);
            }
        }
    }
}

void from_json(const nlohmann::json& jsonFile, Snapshot::SnapshotNode& snapshotNode)
{
    snapshotNode.m_name = jsonFile.value("name", "");
    snapshotNode.m_className = jsonFile.value("classname", "");
    snapshotNode.m_pathName = jsonFile.value("pathname","");

    if (jsonFile.contains("data") && jsonFile["data"].is_array())
    {
        snapshotNode.m_dataContainer.clear();
        for (const auto& dataJson : jsonFile["data"])
        {
            Snapshot::DataInfo dataInfo;
            from_json(dataJson, dataInfo);
            snapshotNode.m_dataContainer.push_back(dataInfo);
        }
    }

    if (jsonFile.contains("links") && jsonFile["links"].is_array())
    {
        snapshotNode.m_linkContainer.clear();
        for (const auto& linkJson : jsonFile["links"])
        {
            Snapshot::LinkInfo linkInfo;
            from_json(linkJson, linkInfo);
            snapshotNode.m_linkContainer.push_back(linkInfo);
        }
    }

    snapshotNode.m_objects.clear();
    if (jsonFile.contains("components") && jsonFile["components"].is_array())
    {
        for (const auto& childJson : jsonFile["components"])
        {
            if (!childJson.is_null())
            {
                auto child = std::make_shared<Snapshot::SnapshotObject>();
                from_json(childJson, *child);
                snapshotNode.m_objects.push_back(child);
            }
        }
    }

    snapshotNode.m_children.clear();
    if (jsonFile.contains("children") && jsonFile["children"].is_array())
    {
        for (const auto& childJson : jsonFile["children"])
        {
            if (!childJson.is_null())
            {
                auto child = std::make_shared<Snapshot::SnapshotNode>();
                from_json(childJson, *child);
                snapshotNode.m_children.push_back(child);
            }
        }
    }
}

namespace jsonsnapshot {
void exportToJSON(const Snapshot &snapshot, const std::string &filename)
{
    nlohmann::ordered_json jsonObject = *snapshot.m_graphRoot;

    std::ofstream file(filename);
    file << jsonObject.dump(5);
    file.close();
}

void importFromJSON(Snapshot &snapshot, const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        msg_error("JSONSnapshot") << "ERROR: Cannot open file " << filename << " for reading";
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
        msg_error("JSONSnapshot") << "Invalid JSON format in " << filename;
        return;
    }

    msg_info("JSONSnapshot") << "JSON imported successfully from: " << filename;
}

std::string fileToString(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        msg_error("JSONSnapshot") << "Cannot open file " << filename << " for reading";
        return "";
    }

    nlohmann::json jsonRoot;
    file >> jsonRoot;
    file.close();
    return to_string(jsonRoot);
}

std::string snapshotToString(const Snapshot &snapshot)
{
    nlohmann::ordered_json jsonString = *snapshot.m_graphRoot;
    return to_string(jsonString);
}

void exportToJSON(const std::map<std::string, std::shared_ptr<Snapshot> > &snapshots, const std::string &filename)
{
    std::ofstream file(filename);

    nlohmann::ordered_json jsonArray = nlohmann::json::array();

    for (const auto &snapshotJson: snapshots)
    {
        jsonArray.push_back(*snapshotJson.second->m_graphRoot);
    }
    file << jsonArray.dump(5);
    file.close();
}

void importFromJSON(std::map<std::string, std::shared_ptr<Snapshot> > &snapshots, const std::string &filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        msg_error("JSONSnapshot") << "Cannot open file " << filename << " for reading";
        return;
    }

    nlohmann::json jsonArray = nlohmann::json::array();
    file >> jsonArray;
    file.close();

    snapshots.clear();

    int index = 0;

    for (const auto &snapshotJson: jsonArray)
    {
        auto snapshot = std::make_shared<Snapshot>();

        from_json(snapshotJson, *snapshot->m_graphRoot);

        std::string id;

        if (snapshot->m_graphRoot)
           id = snapshot->m_graphRoot->m_name;

        if (id.empty())
            id = "snapshot_" + std::to_string(index++);
        snapshot->m_graphRoot->m_name = id;
        snapshots[id] = snapshot;
    }
}

void doLoadSet(const std::string &filename,std::map<std::shared_ptr<sofa::core::objectmodel::Snapshot>, double> &snapshots)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        msg_error("JSONSnapshot") << "Cannot open file " << filename << " for reading";
        return;
    }

    nlohmann::json jsonSnapshot;

    file >> jsonSnapshot;
    file.close();

    for (const auto &snapshotJson: jsonSnapshot)
    {
        auto snapshot = std::make_shared<sofa::core::objectmodel::Snapshot>();
        snapshot->m_graphRoot = std::make_shared<sofa::core::objectmodel::Snapshot::SnapshotNode>();
        sofa::core::objectmodel::from_json(snapshotJson, *snapshot->m_graphRoot);
        std::string snapshotTime = "0";
        for (const auto &data: snapshot->m_graphRoot->m_dataContainer)
        {
            if (data.m_name == "time")
                snapshotTime = data.m_value;
        }
        snapshots.insert({snapshot, std::stod(snapshotTime)});
    }
}
}
} // namespace sofa::core::objectmodel
