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

void SnapshotVisitor::processObject(core::objectmodel::BaseObject* obj, std::string parentName)
{

    obj->saveSnapshot(snapCont_, parentName);
}

Visitor::Result SnapshotVisitor::processNodeTopDown(simulation::Node* node)
{ 
    auto* parent = node->getFirstParent();
    std::string parentName = parent ? parent->getName() : node->getName();
    node->saveSnapshot(snapCont_, parentName);

    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        this->processObject(it->get(),node->getName());
    }
    
    return RESULT_CONTINUE;
}

void SnapshotVisitor::processNodeBottomUp(simulation::Node* /*node*/)
{
}


} // namespace sofa::simulation



