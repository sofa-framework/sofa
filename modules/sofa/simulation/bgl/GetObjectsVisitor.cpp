/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/bgl/GetObjectsVisitor.h>

namespace sofa
{

namespace simulation
{

namespace bgl
{
Visitor::Result GetObjectsVisitor::processNodeTopDown( simulation::Node* node )
{
    if (node != ignoreNode)
        for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
        {
            core::objectmodel::BaseObject* obj = it->get();
            void* result = class_info.dynamicCast(obj);
            if (result != NULL &&  (tags.empty() || (obj)->getTags().includes(tags)))
                container(result);
        }
    return Visitor::RESULT_CONTINUE;
}

Visitor::Result GetObjectVisitor::processNodeTopDown( simulation::Node* node )
{
    if (node != ignoreNode)
        for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
        {
            core::objectmodel::BaseObject* obj = it->get();
            void* r = class_info.dynamicCast(obj);
            if (r != NULL &&  (tags.empty() || (obj)->getTags().includes(tags)))
            {
                result=r; return Visitor::RESULT_PRUNE;
            }
        }
    return Visitor::RESULT_CONTINUE;
}

}
}
}
