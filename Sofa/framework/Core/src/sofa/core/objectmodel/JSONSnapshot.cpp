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
    j["pathname"] = di.pathname;
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
    DataInfo dinfo;
    for (auto* data : datafield)
    {
        dinfo.name = data->getName();
        dinfo.type = data->getValueTypeString();
        dinfo.value = data->getValueString();
        dinfo.pathname = data->getPathName();
        SparseDataSnapshot_.dataContainer.push_back(dinfo); 
    }
    LinkInfo linfo;
    for (auto* link : linkfield)
    {
        linfo.name = link->getName();
        linfo.linkedpath = link->getLinkedPath();
        linfo.path = link->getPath();
        SparseDataSnapshot_.linkContainer.push_back(linfo);
    }
}

void JSONSnapshot::exportToJSON(const std::string filename)
{
    nlohmann::json j = SparseDataSnapshot_;
    std::ofstream file(filename);
    file << j.dump(5);
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

void JSONSnapshot::importFromJSON(const std::string filename, nlohmann::json& j)
{
    std::ifstream file(filename);
    file >> j;
}

void JSONSnapshot::putData(std::vector<BaseData*>& datafield, std::vector<BaseLink*>& linkfield)
{
    // idea -> while-loop with conditions like : while dataname are same, just set the data. if not 
    DataInfo dinfo;
    for (auto* data : datafield)
    {
        dinfo.name = data->getName();
        dinfo.type = data->getValueTypeString();
        dinfo.value = data->getValueString();
        SparseDataSnapshot_.dataContainer.push_back(dinfo); 
    }
    LinkInfo linfo;
    for (auto* link : linkfield)
    {
        linfo.name = link->getName();
        linfo.linkedpath = link->getLinkedPath();
        linfo.path = link->getPath();
        SparseDataSnapshot_.linkContainer.push_back(linfo);
    }
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