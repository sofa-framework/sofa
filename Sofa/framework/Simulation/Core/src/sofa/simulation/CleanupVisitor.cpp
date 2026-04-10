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
#include <sofa/simulation/CleanupVisitor.h>
#include <sofa/simulation/Node.h>


namespace sofa::simulation
{


simulation::Visitor::Result CleanupVisitor::processNodeTopDown(Node* node)
{
    // some object will modify the graph during cleanup (removing other nodes or objects)
    // so we cannot assume that the list of object will stay constant

    std::set<sofa::core::objectmodel::BaseComponent*> done; // list of objects we already processed
    bool stop = false;
    while (!stop)
    {
        stop = true;
        std::vector< core::objectmodel::BaseComponent* > listObject;
        node->get<core::objectmodel::BaseComponent>(&listObject, core::objectmodel::BaseContext::Local);

        for (unsigned int i=0; i<listObject.size(); ++i)
        {
            if (done.insert(listObject[i]).second)
            {
                listObject[i]->cleanup();
                stop = false;
                break; // we have to restart as objects could have been removed anywhere
            }
        }
    }
    return RESULT_CONTINUE;
}

void CleanupVisitor::processNodeBottomUp(Node* /*node*/)
{
}

} // namespace sofa::simulation



