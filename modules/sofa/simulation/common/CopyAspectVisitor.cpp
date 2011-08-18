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

#include "CopyAspectVisitor.h"

namespace sofa
{

namespace simulation
{

CopyAspectVisitor::CopyAspectVisitor(const core::ExecParams* params, int destAspect, int srcAspect)
    : Visitor(params), destAspect(destAspect), srcAspect(srcAspect)
{
}

CopyAspectVisitor::~CopyAspectVisitor()
{
}

CopyAspectVisitor::Result CopyAspectVisitor::processNodeTopDown(Node* node)
{
    node->copyAspect(destAspect, srcAspect);
    for(Node::ObjectIterator iObj = node->object.begin(), endObj = node->object.end(); iObj != endObj; ++iObj)
    {
        fprintf(stderr, "Copy node: %s, object: %s\n", node->getName().c_str(), (*iObj)->getName().c_str());
        (*iObj)->copyAspect(destAspect, srcAspect);
    }
    for(Node::ChildIterator iVNode = node->childInVisualGraph.begin(), endVNode = node->childInVisualGraph.end(); iVNode != endVNode; ++iVNode)
    {
        processNodeTopDown(*iVNode);
    }
    return RESULT_CONTINUE;
}

} // namespace sofa

} // namespace simulation
