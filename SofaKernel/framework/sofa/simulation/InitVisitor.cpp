/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/simulation/Simulation.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/BoundingBox.h>

//#include "MechanicalIntegration.h"

namespace sofa
{

namespace simulation
{


Visitor::Result InitVisitor::processNodeTopDown(simulation::Node* node)
{
    if (!rootNode) rootNode=node;

    node->initialize();
#ifdef SOFA_SMP_NUMA
    if(node->getProcessor()!=-1)
    {
        msg_info()<<"set preferred cpu "<<node->getProcessor()/2<<std::endl;
        numa_set_preferred(node->getProcessor()/2);
    }
#endif

    sofa::defaulttype::BoundingBox* nodeBBox = node->f_bbox.beginEdit(params);
    nodeBBox->invalidate();

    for(unsigned int i=0; i<node->object.size(); ++i)
    {
        node->object[i]->init();
        node->object[i]->computeBBox(params, true);
        nodeBBox->include(node->object[i]->f_bbox.getValue(params));
    }
    node->f_bbox.endEdit(params);
    return RESULT_CONTINUE;
}


void InitVisitor::processNodeBottomUp(simulation::Node* node)
{
    // init all the components in reverse order
    node->setDefaultVisualContextValue();
    sofa::defaulttype::BoundingBox* nodeBBox = node->f_bbox.beginEdit(params);

    for(unsigned int i=node->object.size(); i>0; --i)
    {
        node->object[i-1]->bwdInit();
        nodeBBox->include(node->object[i-1]->f_bbox.getValue(params));
    }

    node->f_bbox.endEdit(params);
    node->bwdInit();
}



} // namespace simulation

} // namespace sofa

