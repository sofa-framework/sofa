/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "ReleaseAspectVisitor.h"

namespace sofa
{

namespace simulation
{

ReleaseAspectVisitor::ReleaseAspectVisitor(const core::ExecParams* params, int aspect)
    : Visitor(params), aspect(aspect)
{
}

ReleaseAspectVisitor::~ReleaseAspectVisitor()
{
}

void ReleaseAspectVisitor::processObject(sofa::core::objectmodel::BaseObject* obj)
{
    obj->releaseAspect(aspect);
    const sofa::core::objectmodel::BaseObject::VecSlaves& slaves = obj->getSlaves();

    for(sofa::core::objectmodel::BaseObject::VecSlaves::const_iterator iObj = slaves.begin(), endObj = slaves.end(); iObj != endObj; ++iObj)
    {
        processObject(iObj->get());
    }
}

ReleaseAspectVisitor::Result ReleaseAspectVisitor::processNodeTopDown(Node* node)
{
    node->releaseAspect(aspect);
    for(Node::ObjectIterator iObj = node->object.begin(), endObj = node->object.end(); iObj != endObj; ++iObj)
    {
        processObject(iObj->get());
    }
    return RESULT_CONTINUE;
}

} // namespace sofa

} // namespace simulation
