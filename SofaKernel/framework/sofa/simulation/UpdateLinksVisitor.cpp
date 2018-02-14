/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/simulation/UpdateLinksVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace simulation
{


void UpdateLinksVisitor::processObject(core::objectmodel::BaseObject* obj)
{
    obj->updateLinks();
}

Visitor::Result UpdateLinksVisitor::processNodeTopDown(simulation::Node* node)
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        this->processObject(it->get());
    }

    //TODO(dmarchal): why do we clear the messsage logs when we update the links ?
    //node->clearWarnings(); node->clearOutputs();
    return RESULT_CONTINUE;
}

void UpdateLinksVisitor::processNodeBottomUp(simulation::Node* /*node*/)
{
}

} // namespace simulation

} // namespace sofa

