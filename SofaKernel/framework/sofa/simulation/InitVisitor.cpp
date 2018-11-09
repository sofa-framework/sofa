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
#include <sofa/simulation/InitVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/BoundingBox.h>

#include <sofa/helper/logging/CountingMessageHandler.h>
using sofa::helper::logging::countingmessagehandler::CountingMessageHandler;
#include <sofa/helper/logging/MessageDispatcher.h>
using sofa::helper::logging::MessageDispatcher;
using sofa::helper::logging::Message;
using sofa::core::objectmodel::ComponentState;

//#include "MechanicalIntegration.h"

namespace sofa
{

namespace simulation
{


Visitor::Result InitVisitor::processNodeTopDown(simulation::Node* node)
{
    CountingMessageHandler* counter = new CountingMessageHandler();
    int isInvalid = counter->getMessageCountFor(Message::Error);
    MessageDispatcher::addHandler(counter);
    if (!rootNode) rootNode=node;

    node->initialize();

    sofa::defaulttype::BoundingBox* nodeBBox = node->f_bbox.beginEdit(params);
    if(!node->f_bbox.isSet())
        nodeBBox->invalidate();

    for(unsigned int i=0; i<node->object.size(); ++i)
    {
        node->object[i]->init();
        node->object[i]->computeBBox(params, true);
        nodeBBox->include(node->object[i]->f_bbox.getValue(params));
    }
    node->f_bbox.endEdit(params);
    MessageDispatcher::rmHandler(counter);
    isInvalid = counter->getMessageCountFor(Message::Error) - isInvalid;
    delete counter;

    if (isInvalid)
        node->setComponentState(ComponentState::Invalid);
    else
        node->setComponentState(ComponentState::Valid);

    return RESULT_CONTINUE;
}


void InitVisitor::processNodeBottomUp(simulation::Node* node)
{
    CountingMessageHandler* counter = new CountingMessageHandler();
    int isInvalid = counter->getMessageCountFor(Message::Error);
    MessageDispatcher::addHandler(counter);

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

    MessageDispatcher::rmHandler(counter);
    isInvalid = counter->getMessageCountFor(Message::Error) - isInvalid;
    delete counter;

    if (isInvalid)
        node->setComponentState(ComponentState::Invalid);
    else
        node->setComponentState(ComponentState::Valid);

}



} // namespace simulation

} // namespace sofa

