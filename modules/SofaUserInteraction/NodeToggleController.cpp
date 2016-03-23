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
#include <sofa/simulation/common/DeactivatedNodeVisitor.h>
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
    if(f_printLog.getValue())
    {
        Node* context = down_cast<Node>(getContext());
        std::cout<<"context name = "<<context->name<<std::endl;

        if(d_key.getValue()=='*')
            std::cout<<"key for toggle = A (default)"<<std::endl;
        else
            std::cout<<"key for toggle = "<<d_key.getValue()<<std::endl;


        if(d_nameNode.getValue()!="")
            std::cout<<"name of specific node = "<<d_nameNode.getValue()<<std::endl;
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

                if(f_printLog.getValue())
                    std::cout<<"specific node found"<<std::endl;
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

                if(f_printLog.getValue())
                    std::cout<<"Activate"<<std::endl;
            }
            else
            {
                n->is_activated.setValue(true);
                sofa::simulation::DeactivationVisitor visitorOFF(sofa::core::ExecParams::defaultInstance(), false);
                n->executeVisitor(&visitorOFF);
                n->setActive(false);

                if(f_printLog.getValue())
                    std::cout<<"Desactivate"<<std::endl;
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

                if(f_printLog.getValue())
                    std::cout<<"Activate"<<std::endl;
            }
            // init is in-active
            else
            {
                specificNode->is_activated.setValue(true);
                sofa::simulation::DeactivationVisitor visitorOFF(sofa::core::ExecParams::defaultInstance(), false);
                specificNode->executeVisitor(&visitorOFF);
                specificNode->setActive(false);

                if(f_printLog.getValue())
                    std::cout<<"Desactivate"<<std::endl;
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

                if(f_printLog.getValue())
                    std::cout<<"Desactivate"<<std::endl;
            }
            else
            {
                specificNode->is_activated.setValue(true);
                specificNode->setActive(true);
                sofa::simulation::DeactivationVisitor visitorON(sofa::core::ExecParams::defaultInstance(), true);
                specificNode->executeVisitor(&visitorON);

                if(f_printLog.getValue())
                    std::cout<<"Activate"<<std::endl;
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
        std::cout << "NodeToggleController: switching active node" << std::endl;
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
            if(f_printLog.getValue())
                std::cout << "NodeToggleController: switching active node" << std::endl;
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
            if(f_printLog.getValue())
                std::cout << "NodeToggleController: switching active node" << std::endl;
            toggle();
        }
    }
}

} // namespace controller

} // namespace component

} // namespace sofa
