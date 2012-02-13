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
#include <sofa/component/controller/NodeToggleController.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/simulation/common/Node.h>


namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::simulation::tree;
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

    Node* context = dynamic_cast<Node*>(getContext());
    std::cout<<"context name = "<<context->name<<std::endl;
    Node::Children children = context->getChildren();
    for (unsigned int i=0; i<children.size(); i++)
    {
        Node* n = dynamic_cast<Node*>(children[i]);
        std::cout<<"child = "<<n->name<<std::endl;
        n->setActive(false);
    }

}

void NodeToggleController::toggle()
{
    // de-activate the current node, and activate the following one
    Node* context = dynamic_cast<Node*>(getContext());
    std::cout<<"context name = "<<context->name<<std::endl;
    Node::Children children = context->getChildren();
    if (children.size()==0) return; // no subnode, return directly
    int nodeIndex = -1;
    for (int i=0; i<(int)children.size() && nodeIndex==-1; i++)
    {
        Node* n = dynamic_cast<Node*>(children[i]);
        if (n->isActive()) nodeIndex = i;
    }
    if (nodeIndex==-1) dynamic_cast<Node*>(children[0])->setActive(true);
    else
    {
        dynamic_cast<Node*>(children[nodeIndex])->setActive(false);
        dynamic_cast<Node*>(children[(nodeIndex+1)%children.size()])->setActive(true);
    }
}

void NodeToggleController::onHapticDeviceEvent(core::objectmodel::HapticDeviceEvent *oev)
{
    // toggle on button 2 pressed
    if (oev->getButton() && core::objectmodel::HapticDeviceEvent::Button2Mask)
        toggle();
}


} // namespace controller

} // namespace component

} // namespace sofa
