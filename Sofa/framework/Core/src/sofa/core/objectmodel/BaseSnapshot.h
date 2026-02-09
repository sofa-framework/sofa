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
//#include <sofa/core/objectmodel/Base.h>
// #include <sofa/core/objectmodel/BaseData.h>
// #include <sofa/core/objectmodel/BaseLink.h>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <sofa/core/config.h>

namespace sofa::core::objectmodel
{

class SOFA_CORE_API BaseSnapshot
{
public:
    struct DataInfo
    {
        std::string name;
        std::string type;
        std::string value;
    };

    struct LinkInfo
    {
        std::string name;
        std::string type;
        std::string value;
    };

    struct SnapshotObject
    {
        std::string m_name;
        std::vector<DataInfo> m_dataContainer;
        std::vector<LinkInfo> m_linkContainer;
        //void* m_internalState { nullptr };

        SnapshotObject() = default;
        explicit SnapshotObject(const std::string& name) : m_name(name){}

        virtual ~SnapshotObject() = default;
    };

    struct SnapshotNode : public SnapshotObject
    {
        std::vector<SnapshotObject> components;
        std::vector<std::shared_ptr<SnapshotNode>> children;
        
        SnapshotNode() = default;
        SnapshotNode(const std::string& name) : SnapshotObject(name) {}
        SnapshotNode(const SnapshotObject& obj) : SnapshotObject(obj) {}

        virtual ~SnapshotNode() noexcept = default;
    };

    std::shared_ptr<SnapshotNode> m_graphRoot { nullptr };

    virtual void importSnapshot(const std::string filename) = 0;

    virtual void exportTo(const std::string filename) = 0;
    virtual void importFrom(std::string filename) = 0;

    void printSnapshot() const;

    BaseSnapshot();
    virtual ~BaseSnapshot() = 0;
};
} // namespace sofa::core::objectmodel
