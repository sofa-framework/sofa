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
#include <sofa/simulation/common/StateChangeVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/common/Node.h>

#include <sofa/core/componentmodel/topology/TopologicalMapping.h>
#include <sofa/core/componentmodel/topology/BaseTopology.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/topology/PointSetTopology.h>

namespace sofa
{

namespace simulation
{


using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

using namespace sofa::core::componentmodel::topology;

using namespace sofa::core;

using namespace sofa::component::topology;


void StateChangeVisitor::processStateChange(core::objectmodel::BaseObject* obj)
{
    obj->handleStateChange();
}

Visitor::Result StateChangeVisitor::processNodeTopDown(simulation::Node* node)
{
    // Topological modifications are not propagated to the contact nodes.
    if (node->getContext()->getName() == "contactPoints")
        return RESULT_PRUNE;

    Result res = RESULT_CONTINUE;

    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        if (dynamic_cast<sofa::core::componentmodel::topology::TopologicalMapping*>(*it) !=  NULL)
        {
            res = RESULT_PRUNE; // stop the propagation of state changes
        }
    }

    if (node->getContext()->getMechanicalState() != NULL)
    {
        this->processStateChange(node->getContext()->getMechanicalState());
    }

    return res;
}



} // namespace simulation

} // namespace sofa
