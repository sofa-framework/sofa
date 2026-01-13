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

void JSONSnapshot::printSnapshot()
{
    std::cout << "printJSONSnapshot data : " << snapshot.size() << std::endl;
}

void to_json(nlohmann::json& j, const BaseSnapshot::DataInfo& di )
{
    j.clear();
    j["name"]  = di.name;
    j["type"]  = di.type;
    j["value"] = di.value;
    j["ownername"] = di.ownername;
}

void to_json(nlohmann::json& j, const BaseSnapshot::LinkInfo& li )
{
    j.clear();
    j["name"]       = li.name;
    j["linkedpath"] = li.linkedpath;
    j["path"]       = li.path;
}

void to_json(nlohmann::json& j, const BaseSnapshot::SparseDataSnapshot& sds )
{
    j.clear();
    j["datas"] = sds.dataContainer;
    j["links"] = sds.linkContainer;
}

void JSONSnapshot::collectData(const std::vector<BaseData*>& datafield, const std::vector<BaseLink*>& linkfield)
{
    SparseDataSnapshot_.dataContainer.clear();
    SparseDataSnapshot_.linkContainer.clear();
    DataInfo dinfo;
    for (auto* data : datafield)
    {
        // std::cout << (*data) << std::endl;
        
        dinfo.name = data->getName();
        dinfo.type = data->getValueTypeString();
        dinfo.value = data->getValueString();
        // (*data).setValueString(dinfo.value); 
        // data->setValueString(dinfo.value);

        dinfo.ownername = (data->getOwner())->getName();
        SparseDataSnapshot_.dataContainer.push_back(dinfo); 
    }
    LinkInfo linfo;
    for (auto* link : linkfield)
    {
        // std::cout << (*link) << std::endl;
        linfo.name = link->getName();
        linfo.linkedpath = link->getLinkedPath();
        linfo.path = link->getPath();
        SparseDataSnapshot_.linkContainer.push_back(linfo);
    }
    SparseSnapshot.push_back(SparseDataSnapshot_);
}

nlohmann::json JSONSnapshot::nodeArray()
{
    nlohmann::json jNode = nlohmann::json::array();

    NodeSnapshot.push_back(SparseSnapshot);
    jNode.push_back(NodeSnapshot);
    SparseSnapshot.clear();

    return jNode;
}

void JSONSnapshot::groupComponent()
{
    NodeSnapshot.push_back(SparseSnapshot);
    SparseSnapshot.clear();
}

void JSONSnapshot::exportTo(const std::string filename)
{
    nlohmann::json root = nlohmann::json::object();

    for (size_t ni = 0; ni < NodeSnapshot.size(); ++ni)
    {
        const auto &components = NodeSnapshot[ni];
        nlohmann::json nodeJson = nlohmann::json::object();

        for (size_t ci = 0; ci < components.size(); ++ci)
        {
            const auto &sds = components[ci];
            std::string compName;
            compName = sds.dataContainer.front().ownername;
            nodeJson[compName] = sds;
        }
        root[ComponentSnapshot[ni]] = nodeJson;
    }

    std::ofstream file(filename);
    file << root.dump(5);
    file.close();
}

void JSONSnapshot::importSnapshot()
{
    std::cout << "importSnapshot" << std::endl;
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
    j.at("linkedpath").get_to(li.linkedpath);
    j.at("path").get_to(li.path);
}

void from_json(const nlohmann::json& j,BaseSnapshot::SparseDataSnapshot& sds )
{
    j.at("datas").get_to(sds.dataContainer);
    j.at("links").get_to(sds.linkContainer);
}

void JSONSnapshot::importFrom(const std::string filename, nlohmann::json& j)
{
    std::ifstream file(filename);
    file >> j;
}

void JSONSnapshot::putData(std::vector<BaseData*>& datafield, std::vector<BaseLink*>& linkfield, BaseSnapshot::DataInfo& di)
{
    // idea -> while-loop with conditions like : while dataname are same, just set the data. if not 
    
    std::cout << "hey" << std::endl;
    
}


void JSONSnapshot::fillDataSnapshot(BaseData* dat)
{
    dataSnapshot_.dataContainer.push_back(dat);
}

void JSONSnapshot::fillSnapshot(DataSnapshot datasnap)
{
    snapshot.push_back(datasnap);
}

void JSONSnapshot::fillLinkSnapshot(BaseLink* link)
{
    dataSnapshot_.linkContainer.push_back(link);
}

// void JSONSnapshot::collectData(const std::vector<BaseData*>& datafield, const std::vector<BaseLink*>& componentlinks )
// {
//     for (auto* data : datafield)
//     {
//         fillDataSnapshot(data);
//     }
    
//     for(auto* link : componentlinks)
//     {
//         fillLinkSnapshot(link);
//     }
//     fillSnapshot(dataSnapshot_);
// }

} // namespace sofa::core::objectmodel