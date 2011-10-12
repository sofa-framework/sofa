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
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
using std::cerr;
using std::endl;

namespace sofa
{

namespace simulation
{


void DeleteVisitor::processNodeBottomUp(Node* node)
{
    while (!node->child.empty())
    {
        Node* child = node->child.begin()->get();
        node->removeChild((Node*)child);
        delete child;
    }
    while (!node->object.empty())
    {
        core::objectmodel::BaseObject* object = node->object.begin()->get();
        node->removeObject(object);
        if (object != (core::objectmodel::BaseObject*)getSimulation())
        {
//                    cerr << "DeleteVisitor::processNodeBottomUp, delete component : " << object->getName() << endl;
            delete object;
        }
    }
}

} // namespace simulation

} // namespace sofa

