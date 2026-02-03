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
#include <sofa/simulation/SnapshotVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/collision/Pipeline.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/topology/BaseTopologyObject.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/visual/Shader.h>
#include <sofa/core/visual/VisualManager.h>
//#include <sofa/core/visual/VisualLoop.h>
#include <sofa/core/visual/BaseVisualStyle.h>
#include <sofa/core/topology/BaseMeshTopology.h>
// #include <sofa/core/BaseState.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/SnapshotFactory.h>
using sofa::core::objectmodel::SnapshotType;
#include <iostream>


namespace sofa::simulation
{

void SnapshotVisitor::processObject(
    core::objectmodel::BaseObject* obj,
    std::shared_ptr<core::objectmodel::BaseSnapshot::SnapNode> parent)
{
    std::vector<std::shared_ptr<core::objectmodel::BaseSnapshot::SnapNode>> parents;
    auto snapshot = obj->saveSnapshot(parents);
    parent->components.push_back(*snapshot);
}

Visitor::Result SnapshotVisitor::processNodeTopDown(simulation::Node* node)
{
    const auto parents = node->getParents();

    std::vector<std::shared_ptr<core::objectmodel::BaseSnapshot::SnapNode>> snapshotParents;
    for (auto* p : parents)
    {
        const auto it = m_snapshotNodeMap.find(p);
        if (it != m_snapshotNodeMap.end())
        {
            snapshotParents.push_back(it->second);
        }
        else
        {
            msg_error("SnapshotVisitor") << "Does it happen??";
        }
    }

    auto snapshot = node->saveSnapshot(snapshotParents);
    auto snapNode = std::dynamic_pointer_cast<core::objectmodel::BaseSnapshot::SnapNode>(snapshot);
    if (snapNode)
    {
        m_snapshotNodeMap[node] = snapNode;
    }

    if (m_snapshotContainer.m_graphRoot == nullptr) //root node
    {
        m_snapshotContainer.m_graphRoot = snapNode;
    }

    for (const auto& it : node->object)
    {
        this->processObject(it.get(), snapNode);
    }
    
    return RESULT_CONTINUE;
}

} // namespace sofa::simulation



