/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
//
// C++ Implementation : NodeToggleController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <SofaUserInteraction/NodeToggleController.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/simulation/DeactivatedNodeVisitor.h>
#include <sofa/helper/cast.h>


namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::simulation;
using namespace sofa::core::objectmodel;

SOFA_DECL_CLASS(NodeToggleController)
// Register in the Factory
int NodeToggleControllerClass = core::RegisterObject("Provides a way to switch active one of the children nodes.")
        .add<NodeToggleController>()
        ;


NodeToggleController::NodeToggleController()
    : d_key( initData(&d_key,(char)'*',"key", "Key chosen for toggling the node(s)") )
    , d_nameNode( initData(&d_nameNode,(std::string) "","nameNode", "Name of a specific node to toggle") )
    , d_initStatus( initData(&d_initStatus,(bool) true,"initStatus", "If one node is chosen, this gives the initial status of the node") )
    , d_firstFrame( initData(&d_firstFrame,(bool) true,"firstFrame", "Toggle the node at first step") )
{
    nodeFound = false;
}


void NodeToggleController::init()
{
    if(notMuted())
    {
        std::stringstream tmp;

        Node* context = down_cast<Node>(getContext());

        tmp<<"context name = "<<context->name<<msgendl;

        if(d_key.getValue()=='*')
            tmp<<"key for toggle = A (default)"<<msgendl;
        else
            tmp<<"key for toggle = "<<d_key.getValue()<<msgendl;

        if(d_nameNode.getValue()!="")
            tmp<<"name of specific node = "<<d_nameNode.getValue();

        msg_info() << tmp.str() ;
    }

    if(d_nameNode.getValue()!="")
    {
        Node* context = down_cast<Node>(getContext());
        Node::Children children = context->getChildren();
        if (children.size()==0) return; // no subnode, return directly
        for (int i=0; i<(int)children.size(); i++)
        {
            Node* n = down_cast<Node>(children[i]);
            if(n->getName() == d_nameNode.getValue())
            {
                specificNode = down_cast<Node>(children[i]);
                nodeFound = true;

                msg_info() << "specific node found" ;
                break;
            }
        }
    }
}

void NodeToggleController::toggle()
{
    if(d_nameNode.getValue()=="")
    {
        Node* context = down_cast<Node>(getContext());
        Node::Children children = context->getChildren();

        if (children.size()==0) return; // no subnode, return directly

        for (int i=0; i<(int)children.size(); i++)
        {
            Node* n = down_cast<Node>(children[i]);

            if (!n->isActive())
            {
                n->is_activated.setValue(true);
                n->setActive(true);
                sofa::simulation::DeactivationVisitor visitorON(sofa::core::ExecParams::defaultInstance(), true);
                n->executeVisitor(&visitorON);

                msg_info() << "Activate" ;
            }
            else
            {
                n->is_activated.setValue(true);
                sofa::simulation::DeactivationVisitor visitorOFF(sofa::core::ExecParams::defaultInstance(), false);
                n->executeVisitor(&visitorOFF);
                n->setActive(false);

                msg_info() << "Desactivate" ;
            }
        }
    }
    else // Case: a specific node is given
    {
        if(down_cast<Node>(getContext())->getChildren().size()==0) return;

        if(d_firstFrame.getValue() && nodeFound)
        {
            // init is active
            if(d_initStatus.getValue())
            {
                specificNode->is_activated.setValue(true);
                specificNode->setActive(true);
                sofa::simulation::DeactivationVisitor visitorON(sofa::core::ExecParams::defaultInstance(), true);
                specificNode->executeVisitor(&visitorON);

                msg_info() << "Activate" ;
            }
            // init is in-active
            else
            {
                specificNode->is_activated.setValue(true);
                sofa::simulation::DeactivationVisitor visitorOFF(sofa::core::ExecParams::defaultInstance(), false);
                specificNode->executeVisitor(&visitorOFF);
                specificNode->setActive(false);

                msg_info() << "Desactivate" ;
            }
        }
        else if(nodeFound)
        {
            if(specificNode->isActive())
            {
                specificNode->is_activated.setValue(true);
                sofa::simulation::DeactivationVisitor visitorOFF(sofa::core::ExecParams::defaultInstance(), false);
                specificNode->executeVisitor(&visitorOFF);
                specificNode->setActive(false);

                msg_info() << "Desactivate" ;
            }
            else
            {
                specificNode->is_activated.setValue(true);
                specificNode->setActive(true);
                sofa::simulation::DeactivationVisitor visitorON(sofa::core::ExecParams::defaultInstance(), true);
                specificNode->executeVisitor(&visitorON);

                msg_info() << "Activate" ;
            }
        }
    }
}


void NodeToggleController::onBeginAnimationStep(const double /*dt*/)
{
    if(d_firstFrame.getValue() && d_initStatus.isSet() && nodeFound)
    {
        toggle();
        d_firstFrame.setValue(false);
    }
}


void NodeToggleController::onHapticDeviceEvent(core::objectmodel::HapticDeviceEvent *oev)
{
    // toggle on button 2 pressed
    if (oev->getButton(1))
    {
        msg_info() << "switching active node" ;
        toggle();
    }
}

void NodeToggleController::onKeyPressedEvent(core::objectmodel::KeypressedEvent *oev)
{
    if(d_key.getValue()=='*')
    {
        switch(oev->getKey())
        {
        case 'A':
        case 'a':
        {
            msg_info() << "switching active node" ;
            toggle();
            break;
        }
        default:
            break;
        }
    }
    else
    {
        if(d_key.getValue()==oev->getKey())
        {
            msg_info() << "switching active node" ;
            toggle();
        }
    }
}

} // namespace controller

} // namespace component

} // namespace sofa
