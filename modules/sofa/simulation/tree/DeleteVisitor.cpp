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
#include <sofa/simulation/tree/DeleteVisitor.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

simulation::Visitor::Result DeleteVisitor::processNodeTopDown(GNode* node)
{
    // some object will modify the graph during cleanup (removing other nodes or objects)
    // so we cannot assume that the list of object will stay constant

    std::set<sofa::core::objectmodel::BaseObject*> done; // list of objects we already processed
    bool stop = false;
    while (!stop)
    {
        stop = true;
        for (GNode::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
            if (done.insert(*it).second)
            {
                (*it)->cleanup();
                stop = false;
                break; // we have to restart as objects could have been removed anywhere
            }
    }
    return RESULT_CONTINUE;
}

void DeleteVisitor::processNodeBottomUp(GNode* node)
{
    while (!node->child.empty())
    {
        GNode* child = *node->child.begin();
        node->removeChild(child);
        delete child;
    }
    while (!node->object.empty())
    {
        core::objectmodel::BaseObject* object = *node->object.begin();
        node->simulation::Node::removeObject(object);
        delete object;
    }
}

} // namespace tree

} // namespace simulation

} // namespace sofa

