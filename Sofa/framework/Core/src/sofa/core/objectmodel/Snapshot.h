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
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <sofa/core/config.h>

#include <sofa/core/objectmodel/Data.h>

namespace sofa::core::objectmodel
{

/**
*  \brief Class for snapshot
*
*  This class contains the structure for a snapshot of a simulation in SOFA.
*  The snapshot contains datas and links, and keep to shape of a scene graph
*/

class SOFA_CORE_API Snapshot
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


        virtual void clear()
        {
            m_dataContainer.clear();
            m_linkContainer.clear();
        }

        void push_back(const DataInfo& data)
        {
            m_dataContainer.push_back(data);
        }

        void push_back(const LinkInfo& link)
        {
            m_linkContainer.push_back(link);
        }

        SnapshotObject() = default;
        explicit SnapshotObject(std::string  name) : m_name(std::move(name)){}

        virtual ~SnapshotObject() = default;
    };

    struct SnapshotNode : public SnapshotObject
    {
        std::vector<SnapshotObject> components;
        std::vector<std::shared_ptr<SnapshotNode>> children;

        void clear() override
        {
            components.clear();
            children.clear();  
            m_dataContainer.clear();
            m_linkContainer.clear();
        }

        void push_back(const std::shared_ptr<SnapshotNode>& child)
        {
            children.push_back(child);
        }

        void push_back(const SnapshotObject& component)
        {
            components.push_back(component);
        }

        SnapshotNode() = default;
        explicit SnapshotNode(const std::string& name) : SnapshotObject(name) {}
        explicit SnapshotNode(const SnapshotObject& obj) : SnapshotObject(obj) {}

        ~SnapshotNode() noexcept override = default;
    };

    std::shared_ptr<SnapshotNode> m_graphRoot { nullptr };

    Snapshot();
    virtual ~Snapshot();
};


} // namespace sofa::core::objectmodel
