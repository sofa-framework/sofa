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
#include <sofa/simulation/common/UpdateBoundingBoxVisitor.h>
#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{
namespace simulation
{

using std::cerr;
using std::endl;

UpdateBoundingBoxVisitor::UpdateBoundingBoxVisitor(const sofa::core::ExecParams* params)
    :Visitor(params)
{

}

Visitor::Result UpdateBoundingBoxVisitor::processNodeTopDown(Node* node)
{
    using namespace sofa::core::objectmodel;
    helper::vector<BaseObject*> objectList;
    helper::vector<BaseObject*>::iterator object;
    node->get<BaseObject>(&objectList,BaseContext::Local);
    sofa::defaulttype::BoundingBox* nodeBBox = node->f_bbox.beginEdit(params);
    nodeBBox->invalidate();
    for ( object = objectList.begin(); object != objectList.end(); ++object)
    {
        (*object)->computeBBox(params);
//        cerr<<"UpdateBoundingBoxVisitor::processNodeTopDown object " << (*object)->getName() << " = "<< (*object)->f_bbox.getValue(params) << endl;
        nodeBBox->include((*object)->f_bbox.getValue(params));
//        cerr << "   new bbox = " << *nodeBBox << endl;
    }
    node->f_bbox.endEdit(params);
    return RESULT_CONTINUE;
}

void UpdateBoundingBoxVisitor::processNodeBottomUp(simulation::Node* node)
{
    sofa::defaulttype::BoundingBox* nodeBBox = node->f_bbox.beginEdit(params);
    Node::ChildIterator childNode;
    for( childNode = node->child.begin(); childNode!=node->child.end(); ++childNode)
    {
//        cerr<<"   UpdateBoundingBoxVisitor::processNodeBottomUpDown object " << (*childNode)->getName() << endl;
        nodeBBox->include((*childNode)->f_bbox.getValue(params));
    }
    node->f_bbox.endEdit(params);
}

}
}
