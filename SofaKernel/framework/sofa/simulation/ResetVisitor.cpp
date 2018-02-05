/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/simulation/ResetVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace simulation
{


void ResetVisitor::processObject(core::objectmodel::BaseObject* obj)
{
    obj->reset();
}

Visitor::Result ResetVisitor::processNodeTopDown(simulation::Node* node)
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        this->processObject(it->get());
    }

    return RESULT_CONTINUE;
}

void ResetVisitor::processNodeBottomUp(simulation::Node* /*node*/)
{
}

void StoreResetStateVisitor::processObject(core::objectmodel::BaseObject* obj)
{
    obj->storeResetState();
}

Visitor::Result StoreResetStateVisitor::processNodeTopDown(simulation::Node* node)
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        this->processObject(it->get());
    }
    return RESULT_CONTINUE;
}

void StoreResetStateVisitor::processNodeBottomUp(simulation::Node* /*node*/)
{
}

} // namespace simulation

} // namespace sofa

