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
#pragma once
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/BaseLink.h>

namespace sofa::core::objectmodel
{
class Base;
class SOFA_CORE_API BaseSnapshot 
{
    
public:
    struct DataInfo
    {
        std::string name;
        std::string type;
        std::string value;
        
        DataInfo() : name(), type(), value() {}
    };

    struct LinkInfo
    {
        std::string name;
        std::string type;
        std::string value;
        LinkInfo() : name(), type(), value() {}
    };

    struct SnapComponent
    {
        std::string name;
        std::vector<DataInfo> dataContainer;
        std::vector<LinkInfo> linkContainer;
        
        SnapComponent() : name(""), dataContainer(), linkContainer() {}

        SnapComponent(const std::string& name) : name(name){}
    };

    

    struct SnapNode
    {
        std::string name;
        std::vector<DataInfo> dataContainer;
        std::vector<LinkInfo> linkContainer;
        std::vector<SnapComponent> components;
        std::vector<std::shared_ptr<SnapNode>> children;
        
        SnapNode() : name(""), dataContainer(), linkContainer(), components(), children() {}

        SnapNode(const std::string& name) : name(name) {}
    };

    public :
        SnapComponent SnapComponent_;
        SnapNode SnapNode_;

        std::vector<std::string> nodeList;
        std::vector<std::shared_ptr<SnapNode>> treeSnapshot; // here, it is the snapshot
    

public:
    virtual void importSnapshot(const std::string filename) = 0;

    void printSnapshot();
    void addToSnap(Base& b, SnapNode& snapObj);
    void addToSnap(Base& b, SnapComponent& snapObj);
    std::vector<DataInfo> collectSnapData(const std::vector<BaseData*>& datafield);
    std::vector<LinkInfo> collectSnapLink(const std::vector<BaseLink*>& linkfield);
    bool hasSnapParent(std::string& parentName);
    std::shared_ptr<SnapNode> getSnapParent(std::shared_ptr<SnapNode>& node, std::string& parentName);

    void addToSimulation();

    virtual void exportTo(const std::string filename) = 0;
    virtual void importFrom(std::string filename) = 0;

    BaseSnapshot();
    virtual ~BaseSnapshot() = 0;
};
} // namespace sofa::core::objectmodel