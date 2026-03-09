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
#include <sofa/simulation/LoadDataSnapshotVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/objectmodel/SnapshotFactory.h>
using sofa::core::objectmodel::SnapshotType;

namespace sofa::simulation
{

void LoadDataSnapshotVisitor::processObject(
    core::objectmodel::BaseObject* obj,
    const std::shared_ptr<core::objectmodel::BaseSnapshot::SnapshotNode>& parent
)
{
    auto snapshotObject = obj->findSnapshotObject(parent, obj->getName());
    obj->loadDataSnapshot(snapshotObject);
}

Visitor::Result LoadDataSnapshotVisitor::processNodeTopDown(simulation::Node* node)
{
    const auto snapshotObject = node->findSnapshotObject(m_snapshotContainer.m_graphRoot, node->getName());

    const auto SnapshotNode = std::dynamic_pointer_cast<core::objectmodel::BaseSnapshot::SnapshotNode>(snapshotObject);
    node->loadDataSnapshot(SnapshotNode);
    std::cout << "test object.getSize : " << node->object.getSize() << std::endl;
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        this->processObject(it->get(), SnapshotNode);
    }
    return RESULT_CONTINUE;
}

} // namespace sofa::simulation



