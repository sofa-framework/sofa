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

void to_json(nlohmann::json& j, const BaseSnapshot::SnapComponent& sds )
{
    j.clear();
    j["datas"] = sds.dataContainer;
    j["links"] = sds.linkContainer;
}

void to_json(nlohmann::json& j, const BaseSnapshot::SnapNode& sn)
{
    j.clear();
    j["name"] = sn.name;
    j["datas"] = sn.dataContainer;
    j["links"] = sn.linkContainer;
    j["componentList"] = sn.componentList;

    j["childNode"] = nlohmann::json::array();
    for (const auto& childPtr : sn.childNode)
    {
        if(childPtr)
        {
            j["childNode"].push_back(*childPtr);
        }
    }
}


std::shared_ptr<BaseSnapshot::SnapNode> JSONSnapshot::createChildNode(const std::string& name)
{
    auto child = std::make_shared<SnapNode>();
    child->name = name;
    return child;
}

void JSONSnapshot::addChildToCurrentNode(std::shared_ptr<BaseSnapshot::SnapNode> child, BaseSnapshot::SnapNode& snapnode)
{
    // snapnode.childNode.push_back(child);
    // auto currentNode = getCurrentNode();
    // if (currentNode && child)
    //     currentNode->childNode.push_back(child);
    std::cout << "wip" << std::endl;
}


void JSONSnapshot::exportTo(const std::string filename)
{

    nlohmann::json j = nlohmann::json::array() ;

    for (const auto& nodePtr : treeSnapshot)
    {
        if (nodePtr)
        {
            j.push_back(*nodePtr); 
        }
    }

    std::ofstream file(filename);
    file << j.dump(5);
    file.close();
}

void JSONSnapshot::importSnapshot(const std::string filename)
{
    std::cout << "importSnapshot" << std::endl;
    //importFrom(filename);
    
}


void from_json(const nlohmann::json& j,BaseSnapshot::DataInfo& di )
{
    j.at("name").get_to(di.name);
    j.at("type").get_to(di.type);
    j.at("value").get_to(di.value);
    
}

void from_json(const nlohmann::json& j,BaseSnapshot::LinkInfo& li )
{
    j.at("name").get_to(li.name);
    j.at("type").get_to(li.type);
    j.at("value").get_to(li.value);
}

void from_json(const nlohmann::json& j,BaseSnapshot::SnapComponent& sds )
{
    j.at("datas").get_to(sds.dataContainer);
    j.at("links").get_to(sds.linkContainer);
}

void from_json(const nlohmann::json& j, BaseSnapshot::SnapNode& sn)
{
    // j.clear();
    j.at("name").get_to(sn.name); //j["name"] = sn.name;
    j.at("datas").get_to(sn.dataContainer); //j["datas"] = sn.dataContainer;
    j.at("links").get_to(sn.linkContainer); //j["links"] = sn.linkContainer;
    j.at("componentList").get_to(sn.componentList); //j["componentList"] = sn.componentList;

    sn.childNode.clear();
    if (j.contains("childNode") && j["childNode"].is_array())
    {
        for (const auto& childJson : j["childNode"])
        {
            auto child = std::make_shared<BaseSnapshot::SnapNode>();
            childJson.get_to(*child);
            sn.childNode.push_back(child);
        }
    }
}

void JSONSnapshot::importFrom(const std::string filename, BaseSnapshot::SnapNode& rootNode)
{
    std::cout << "importFrom" << std::endl;
    std::ifstream file(filename);
    if(file.is_open())
    {
        nlohmann::json jfile;
        file >> jfile;
        file.close();

        // Lines above would be useful for unit tests 

        // for (auto& [key,value] : data.items())
        // {
        //     std::cout << "name : " << data.value("name","") << std::endl;
        //     std::cout << "key : " << key << std::endl;
        // }

        // if(jfile.contains("componentList"))
        // {
        //     std::cout << "componentList" << std::endl;
        //     for(auto& [componentkey,componentvalue] : jfile["componentList"].items())
        //     {
                
        //         std::cout << "Component : " << componentvalue["datas"][0].value("value","") << std::endl;
        //         std::cout << "Datas : " << std::endl;
        //         for(const auto& d : componentvalue["datas"])
        //         {
        //             std::cout << "  - " << d.value("name", "") << " [" << d.value("type", "") << "] = " << d.value("value", "") << std::endl;
        //         }
        //         std::cout << "Links : " << std::endl;
        //         for(const auto& l : componentvalue["links"])
        //         {
        //             std::cout << " - " << l.value("name","") << ", value : "<<l.value("value","") << std::endl;
        //         }
        //     }
        // }
        
        // SnapNodeContainer = data.get<std::vector<std::vector<_>>>();
        // std::cout << data.dump(2) << std::endl;
    }
}




} // namespace sofa::core::objectmodel