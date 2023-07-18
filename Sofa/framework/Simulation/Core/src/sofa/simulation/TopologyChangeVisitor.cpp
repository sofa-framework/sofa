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

#include <sofa/simulation/TopologyChangeVisitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/core/topology/Topology.h>

namespace sofa
{

namespace simulation
{


using namespace sofa::defaulttype;
using namespace sofa::core::behavior;

using namespace sofa::core::topology;

using namespace sofa::core;


TopologyChangeVisitor::TopologyChangeVisitor(const sofa::core::ExecParams* params, sofa::core::topology::Topology* source)
    : Visitor(params)
    , m_source(source)
{

}

std::string TopologyChangeVisitor::getInfos() const
{
    return "Topology:" + m_source->getName();
}

void TopologyChangeVisitor::processTopologyChange(simulation::Node *node, sofa::core::objectmodel::BaseObject* obj)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment("processTopologyChange");
#endif

    const ctime_t t0=begin(node,obj);
    obj->handleTopologyChange(m_source); //why was it necessary to check for each object if it exists a topology inside the current node?
    end(node,obj,t0);
}


Visitor::Result TopologyChangeVisitor::processNodeTopDown(simulation::Node* node)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment("updateTopologicalMappingTopDown");
#endif

    // search for topological mapping, run the mapping first. Do not stop the propagation of parent topology
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        sofa::core::topology::TopologicalMapping* obj = dynamic_cast<sofa::core::topology::TopologicalMapping*>(it->get());
        if (obj != nullptr)  // find a TopologicalMapping node among the brothers (it must be the first one written)
        {
            if(obj->propagateFromInputToOutputModel() && obj->getFrom() == m_source)  //node != root){ // the propagation of topological changes comes (at least) from a father node, not from a brother
            {
                const ctime_t t0=begin(node,obj);
                obj->updateTopologicalMappingTopDown(); // update the specific TopologicalMapping
                end(node,obj,t0);
            }
        }
    }

    // loop on all node to call handleTopologyChange
    for_each(this, node, node->object,  &TopologyChangeVisitor::processTopologyChange);

    return RESULT_CONTINUE;
}


void TopologyChangeVisitor::processNodeBottomUp(simulation::Node* node)
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        sofa::core::topology::TopologicalMapping* obj = dynamic_cast<sofa::core::topology::TopologicalMapping*>(it->get());
        if (obj != nullptr)  // find a TopologicalMapping node among the brothers (it must be the first one written)
        {
            if(obj->propagateFromOutputToInputModel() && obj->getTo() == m_source)  //node == root){
            {
                obj->updateTopologicalMappingBottomUp(); // update the specific TopologicalMapping
            }
        }
    }
}


} // namespace simulation

} // namespace sofa

