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
#include <sofa/simulation/InitVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/type/BoundingBox.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace simulation
{


Visitor::Result InitVisitor::processNodeTopDown(simulation::Node* node)
{
    if (!rootNode) rootNode=node;

    node->initialize();

    sofa::type::BoundingBox* nodeBBox = node->f_bbox.beginEdit();
    if(!node->f_bbox.isSet())
        nodeBBox->invalidate();

    for(unsigned int i=0; i<node->object.size(); ++i)
    {
        node->object[i]->init();
        node->object[i]->computeBBox(params, true);
        nodeBBox->include(node->object[i]->f_bbox.getValue());
    }
    node->f_bbox.endEdit();
    return RESULT_CONTINUE;
}


void InitVisitor::processNodeBottomUp(simulation::Node* node)
{
    // init all the components in reverse order
    node->setDefaultVisualContextValue();
    sofa::type::BoundingBox* nodeBBox = node->f_bbox.beginEdit();

    for(std::size_t i=node->object.size(); i>0; --i)
    {
        node->object[i-1]->bwdInit();
        nodeBBox->include(node->object[i-1]->f_bbox.getValue());
    }

    node->f_bbox.endEdit();
}



} // namespace simulation

} // namespace sofa

