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
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/Node.h>


namespace sofa::simulation
{


void DeleteVisitor::processNodeBottomUp(Node* node)
{
    while (!node->child.empty())
    {
        Node::SPtr child = *node->child.begin();
        node->removeChild(child);
        child.reset();
    }
    while (!node->object.empty())
    {
        core::objectmodel::BaseObject::SPtr object = *node->object.begin();
        node->removeObject(object);
        object.reset();
    }
}

} // namespace sofa::simulation



