/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/common/ResetVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace simulation
{


void ResetVisitor::processObject(core::objectmodel::BaseObject* obj)
{
    obj->reset();
    //obj->clearWarnings(); obj->clearOutputs();
}

Visitor::Result ResetVisitor::processNodeTopDown(simulation::Node* node)
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        this->processObject(*it);
    }

    {
        if (!node->nodeInVisualGraph.empty()) node->nodeInVisualGraph->execute<ResetVisitor>(this->params);
        for (simulation::Node::ChildIterator itChild = node->childInVisualGraph.begin(); itChild != node->childInVisualGraph.end(); ++itChild)
        {
            simulation::Node *child=*itChild;
            child->execute<ResetVisitor>(this->params);
        }
    }

    node->clearWarnings(); node->clearOutputs();
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
        this->processObject(*it);
    }

    {
        if (!node->nodeInVisualGraph.empty()) node->nodeInVisualGraph->execute<StoreResetStateVisitor>(this->params);
        for (simulation::Node::ChildIterator itChild = node->childInVisualGraph.begin(); itChild != node->childInVisualGraph.end(); ++itChild)
        {
            simulation::Node *child=*itChild;
            child->execute<StoreResetStateVisitor>(this->params);
        }
    }
    return RESULT_CONTINUE;
}

void StoreResetStateVisitor::processNodeBottomUp(simulation::Node* /*node*/)
{
}

} // namespace simulation

} // namespace sofa

