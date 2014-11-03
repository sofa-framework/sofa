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
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/DeactivatedNodeVisitor.h>


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

void NodeToggleController::init()
{
    // register all subnodes, de-activate them and toggle the first one
    std::cout<<"NodeToggleController::init()"<<std::endl;

    m_FirstFrame = true;


}

void NodeToggleController::toggle()
{
    // de-activate the current node, and activate the following one
    Node* context = dynamic_cast<Node*>(getContext());
    //std::cout<<"context name = "<<context->name<<std::endl;
    Node::Children children = context->getChildren();
    if (children.size()==0) return; // no subnode, return directly
    int prevNodeIndex = -1;
    int newNodeIndex = -1;
    for (int i=0; i<(int)children.size() && prevNodeIndex==-1; i++)
    {
        Node* n = dynamic_cast<Node*>(children[i]);
        if (n->isActive()) prevNodeIndex = i;
    }
    if (prevNodeIndex==-1)
    {
        int newNodeIndex=0;

        dynamic_cast<Node*>(children[newNodeIndex])->is_activated.setValue(true,true);
        dynamic_cast<Node*>(children[newNodeIndex])->setActive(true);
        sofa::simulation::DeactivationVisitor visitorON(sofa::core::ExecParams::defaultInstance(), true);
        dynamic_cast<Node*>(children[newNodeIndex])->executeVisitor(&visitorON);
        std::cout<<"Activate"<<std::endl;
    }
    else
    {
        //newNodeIndex = (prevNodeIndex+1)%children.size();
        dynamic_cast<Node*>(children[prevNodeIndex])->is_activated.setValue(true,true);
        sofa::simulation::DeactivationVisitor visitorOFF(sofa::core::ExecParams::defaultInstance(), false);
        dynamic_cast<Node*>(children[prevNodeIndex])->executeVisitor(&visitorOFF);
        dynamic_cast<Node*>(children[prevNodeIndex])->setActive(false);
        std::cout<<"Desactivate"<<std::endl;
    }
}


void NodeToggleController::onBeginAnimationStep(const double /*dt*/)
{
    // deactivate all but first sub-nodes
    if (m_FirstFrame)
    {
        m_FirstFrame = false;

        Node* context = dynamic_cast<Node*>(getContext());
        std::cout<<"context name = "<<context->name<<std::endl;
        Node::Children children = context->getChildren();
//        for (unsigned int i=0; i<children.size(); i++)
//        {
//            Node* n = dynamic_cast<Node*>(children[i]);
//            std::cout<<"child = "<<n->name<<std::endl;
//            n->setActive(false);
//            sofa::simulation::DeactivationVisitor v(sofa::core::ExecParams::defaultInstance(), false);
//            n->executeVisitor(&v);
//        }

        toggle();
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
    switch(oev->getKey())
    {
    case 'A':
    case 'a':
    {
        std::cout << "NodeToggleController: switching active node" << std::endl;
        toggle();
        break;
    }
    default:
        break;
    }
}

} // namespace controller

} // namespace component

} // namespace sofa
