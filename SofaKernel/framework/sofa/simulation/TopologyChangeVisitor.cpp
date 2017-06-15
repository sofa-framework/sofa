/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

namespace sofa
{

namespace simulation
{


using namespace sofa::defaulttype;
using namespace sofa::core::behavior;

using namespace sofa::core::topology;

using namespace sofa::core;


void TopologyChangeVisitor::processTopologyChange(simulation::Node *node, core::objectmodel::BaseObject* obj)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment("processTopologyChange");
#endif
    ctime_t t0=begin(node,obj);
    obj->handleTopologyChange(source); //why was it necessary to check for each object if it exists a topology inside the current node?
    end(node,obj,t0);
}
void TopologyChangeVisitor::processTopologyChangeNoCheck(simulation::Node *node, core::objectmodel::BaseObject* obj)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment("processTopologyChangeNoCheck");
#endif
    ctime_t t0=begin(node,obj);
    obj->handleTopologyChange();
    end(node,obj,t0);
}

Visitor::Result TopologyChangeVisitor::processNodeTopDown(simulation::Node* node)
{
    //if (!root) root = node;
    bool is_TopologicalMapping = false;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment("updateTopologicalMappingTopDown");
#endif
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        sofa::core::topology::TopologicalMapping* obj = dynamic_cast<sofa::core::topology::TopologicalMapping*>(it->get());
        if (obj != NULL)  // find a TopologicalMapping node among the brothers (it must be the first one written)
        {

            if(obj->propagateFromInputToOutputModel() && obj->getFrom() == source)  //node != root){ // the propagation of topological changes comes (at least) from a father node, not from a brother
            {

                ctime_t t0=begin(node,obj);
                obj->updateTopologicalMappingTopDown(); // update the specific TopologicalMapping
                is_TopologicalMapping = true;
                end(node,obj,t0);
            }
        }
    }

    if(is_TopologicalMapping)  // find one TopologicalMapping node among the brothers (which must be the first one written in the scene file)
    {
        //return RESULT_PRUNE; // stop the propagation of topological changes
    }

    for_each(this, node, node->object,  &TopologyChangeVisitor::processTopologyChange);



    return RESULT_CONTINUE;
}

void TopologyChangeVisitor::processNodeBottomUp(simulation::Node* node)
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        sofa::core::topology::TopologicalMapping* obj = dynamic_cast<sofa::core::topology::TopologicalMapping*>(it->get());
        if (obj != NULL)  // find a TopologicalMapping node among the brothers (it must be the first one written)
        {

            if(obj->propagateFromOutputToInputModel() && obj->getTo() == source)  //node == root){
            {

                obj->updateTopologicalMappingBottomUp(); // update the specific TopologicalMapping
            }
        }
    }
}

Visitor::Result HandleTopologyChangeVisitor::processNodeTopDown(simulation::Node* node)
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        core::objectmodel::BaseObject* obj=it->get();
        ctime_t t0=begin(node,obj);
        obj->handleTopologyChange();
        end(node,obj,t0);
    }

    return RESULT_CONTINUE;
}



} // namespace simulation

} // namespace sofa

