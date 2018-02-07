/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/Node.h>

namespace sofa
{
namespace simulation
{

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
    if(!node->f_bbox.isSet())
        nodeBBox->invalidate();
    for ( object = objectList.begin(); object != objectList.end(); ++object)
    {
        // warning the second parameter should NOT be false
        // otherwise every object will participate to the bounding box
        // when it makes no sense for some of them
        // e.g. some loaders have different scale that the states displayed and used for the simu
        // sometimes their values do not even have a spatial meaning (such as MechanicalObject representing constraint value)
        // if some objects does not participate to the bounding box where they should,
        // you should overload their computeBBox function to correct that
        (*object)->computeBBox(params, true);
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
