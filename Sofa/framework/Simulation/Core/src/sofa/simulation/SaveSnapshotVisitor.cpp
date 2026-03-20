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
#include <sofa/simulation/SaveSnapshotVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/Node.h>
#include <iostream>


namespace sofa::simulation
{

void SaveSnapshotVisitor::processObject(
    core::objectmodel::BaseObject* obj,
    std::shared_ptr<core::objectmodel::Snapshot::SnapshotNode> parent)
{
    std::vector<std::shared_ptr<core::objectmodel::Snapshot::SnapshotNode>> parents;
    const auto snapshot = obj->saveSnapshot(parents);
    parent->components.push_back(*snapshot);
}

Visitor::Result SaveSnapshotVisitor::processNodeTopDown(simulation::Node* node)
{
    const auto parents = node->getParents();

    std::vector<std::shared_ptr<core::objectmodel::Snapshot::SnapshotNode>> snapshotParents;
    for (auto* p : parents)
    {
        const auto it = m_snapshotNodeMap.find(p);
        if (it != m_snapshotNodeMap.end())
        {
            snapshotParents.push_back(it->second);
        }
    }

    const auto snapshot = node->saveSnapshot(snapshotParents);
    const auto SnapshotNode = std::dynamic_pointer_cast<core::objectmodel::Snapshot::SnapshotNode>(snapshot);
    if (SnapshotNode)
    {
        m_snapshotNodeMap[node] = SnapshotNode;
    }

    if (m_snapshotContainer.m_graphRoot == nullptr)
    {
        m_snapshotContainer.m_graphRoot = SnapshotNode;
    }

    for (const auto& it : node->object)
    {
        this->processObject(it.get(), SnapshotNode);
    }
    
    return RESULT_CONTINUE;
}

} // namespace sofa::simulation



