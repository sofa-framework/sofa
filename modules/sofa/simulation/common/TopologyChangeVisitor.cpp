/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/simulation/common/TopologyChangeVisitor.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/core/componentmodel/topology/TopologicalMapping.h>

namespace sofa
{

namespace simulation
{


using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

using namespace sofa::core::componentmodel::topology;

using namespace sofa::core;


void TopologyChangeVisitor::processTopologyChange(core::objectmodel::BaseObject* obj)
{
    obj->handleTopologyChange();
}

Visitor::Result TopologyChangeVisitor::processNodeTopDown(simulation::Node* node)
{
    bool is_TopologicalMapping = false;

    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        if (dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping*>(*it)!= NULL)  // find a TopologicalMapping node among the brothers (it must be the first one written)
        {

            sofa::core::componentmodel::topology::TopologicalMapping* obj = dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping*>(*it);

            if(getNbIter() > 0)  // the propagation of topological changes comes (at least) from a father node, not from a brother
            {

                obj->updateTopologicalMapping(); // update the specific TopologicalMapping
                is_TopologicalMapping = true;
            }
        }
    }

    if(is_TopologicalMapping)  // find one TopologicalMapping node among the brothers (which must be the first one written in the scene file)
    {

        // Increment the number of iterations of the method processNodeTopDown
        incrNbIter();
        return RESULT_PRUNE; // stop the propagation of topological changes
    }

    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        this->processTopologyChange(*it);
    }

    // Increment the number of iterations of the method processNodeTopDown
    incrNbIter();
    return RESULT_CONTINUE;
}



} // namespace simulation

} // namespace sofa

