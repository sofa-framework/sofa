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


// void from_json(const nlohmann::json& j,BaseSnapshot::DataInfo& di )
// {
//     j.at("name").get_to(di.name);
//     j.at("type").get_to(di.type);
//     j.at("value").get_to(di.value);
    
// }

// void from_json(const nlohmann::json& j,BaseSnapshot::LinkInfo& li )
// {
//     j.at("name").get_to(li.name);
//     j.at("type").get_to(li.type);
//     j.at("value").get_to(li.value);
// }

// void from_json(const nlohmann::json& j,BaseSnapshot::SnapshotObject& so )
// {
//     j.at("datas").get_to(so.m_dataContainer);
//     j.at("links").get_to(so.m_linkContainer);
// }

// void from_json(const nlohmann::json& j, BaseSnapshot::SnapshotNode& sn)
// {
//     j.at("components").get_to(sn.components); 

//     sn.children.clear();
//     if (j.contains("children") && j["children"].is_array())
//     {
//         for (const auto& childJson : j["children"])
//         {
//             auto child = std::make_shared<BaseSnapshot::SnapshotNode>();
//             childJson.get_to(*child);
//             sn.children.push_back(child);
//         }
//     }
// }

void JSONSnapshot::importFrom(const std::string filename)
{
    // std::cout << "importFrom" << std::endl;
    // std::ifstream file(filename);
    // nlohmann::json jfile;
    // file >> jfile;
    // file.close();
    
    // treeSnapshot.clear();
    
    // // std::cout << "root : " << jfile[0]["name"] << std::endl;
    // // std::cout << "children : " << jfile[0]["children"] << std::endl;
    // for (auto& [key,value] : jfile[0].items())
    // {
    //     std::cout << "TEST" << std::endl;
    // }

    std::cout << "JSON imported successfully from: " << filename << std::endl;

}




} // namespace sofa::core::objectmodel