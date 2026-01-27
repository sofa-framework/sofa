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
#include <sofa/core/objectmodel/BaseSnapshot.h>
#include "BaseSnapshot.h"

namespace sofa::core::objectmodel
{

BaseSnapshot::BaseSnapshot()
{}
BaseSnapshot::~BaseSnapshot() = default;


void BaseSnapshot::printSnapshot()
{
    std::cout << "Snapshot : " << std::endl;
    for (auto element : treeSnapshot)
    {
        std::cout << element->name << std::endl;
    }
}


void BaseSnapshot::addToSnap(Base& b, SnapNode& snapObject)
{
    std::vector<DataInfo> objData = collectSnapData(b.getDataFields());
    snapObject.dataContainer.insert(snapObject.dataContainer.end(),std::make_move_iterator(objData.begin()),std::make_move_iterator(objData.end()));
    std::vector<LinkInfo> objLink = collectSnapLink(b.getLinks());
    snapObject.linkContainer.insert(snapObject.linkContainer.end(),std::make_move_iterator(objLink.begin()),std::make_move_iterator(objLink.end()));
    snapObject.name = b.getName();
}

void BaseSnapshot::addToSnap(Base& b, SnapComponent& snapObject)
{
    std::vector<DataInfo> objData = collectSnapData(b.getDataFields());
    snapObject.dataContainer.insert(snapObject.dataContainer.end(),std::make_move_iterator(objData.begin()),std::make_move_iterator(objData.end()));
    std::vector<LinkInfo> objLink = collectSnapLink(b.getLinks());
    snapObject.linkContainer.insert(snapObject.linkContainer.end(),std::make_move_iterator(objLink.begin()),std::make_move_iterator(objLink.end()));
    snapObject.name = b.getName();
}



std::vector<BaseSnapshot::DataInfo> BaseSnapshot::collectSnapData(const std::vector<BaseData*>& datafield)
{
    BaseSnapshot::DataInfo dinfo;
    std::vector<DataInfo> dataContainer;
    for (auto* data : datafield)
    {
        dinfo.name = data->getName();
        dinfo.type = data->getValueTypeString();
        dinfo.value = data->getValueString();
        dataContainer.push_back(dinfo); 
    }
    return dataContainer;
}

std::vector<BaseSnapshot::LinkInfo> BaseSnapshot::collectSnapLink(const std::vector<BaseLink*>& linkfield)
{
    BaseSnapshot::LinkInfo linfo;
    std::vector<LinkInfo> linkContainer;
    for (auto* link : linkfield)
    {
        linfo.name = link->getName();
        linfo.value = link->getValueString();
        linfo.type = link->getValueTypeString();
        linkContainer.push_back(linfo);
    }
    return linkContainer;
}

bool BaseSnapshot::hasSnapParent(std::string& parentName)
{
    bool result = false;

    if(treeSnapshot.empty())
    {
        return result;
    }
    
    for(auto name : nodeList)
    {
        if(name == parentName)
        {
            result = true;
        }
    }

    return result;
}


std::shared_ptr<BaseSnapshot::SnapNode> BaseSnapshot::getSnapParent(std::shared_ptr<BaseSnapshot::SnapNode>& node,
                                                      std::string& parentName)
{
    if (!node)
    {
        return nullptr;
    }
        
    if(node->name == parentName)
    {
        return node;
    }

    for (auto& child : node->children)
    {
        if(auto result = getSnapParent(child, parentName))
        {
            return result;
        }
    }
    return nullptr;
}

void BaseSnapshot::addToSimulation()
{
    std::cout << "addToSimulation" << std::endl;
    for (auto element : treeSnapshot)
    {
        std::cout << element->name << std::endl;
    }
}

} // namespace sofa::core::objectmodel