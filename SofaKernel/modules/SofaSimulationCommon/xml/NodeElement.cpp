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
#include <SofaSimulationCommon/xml/NodeElement.h>
//#include <SofaSimulationCommon/xml/ObjectElement.h>
#include <SofaSimulationCommon/xml/Element.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace simulation
{


namespace xml
{

using namespace sofa::defaulttype;


NodeElement::NodeElement(const std::string& name, const std::string& type, BaseElement* parent)
    : Element<core::objectmodel::BaseNode>(name, type, parent)
{
}

NodeElement::~NodeElement()
{
}

bool NodeElement::setParent(BaseElement* newParent)
{
    if (newParent != NULL && dynamic_cast<NodeElement*>(newParent)==NULL)
        return false;
    else
        return Element<core::objectmodel::BaseNode>::setParent(newParent);
}

bool NodeElement::initNode()
{
    core::objectmodel::BaseNode::SPtr obj = Factory::CreateObject(this->getType(), this);
    if (obj != NULL)
    {
        setObject(obj);
        core::objectmodel::BaseNode* baseNode;
        if (getTypedObject()!=NULL && getParentElement()!=NULL && (baseNode = getParentElement()->getObject()->toBaseNode()))
        {
            baseNode->addChild(getTypedObject());
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool NodeElement::init()
{
    bool res = Element<core::objectmodel::BaseNode>::init();

    /// send the errors created by the object in this node in the node's log
    for (unsigned int i=0; i<errors.size(); ++i)
    {
        msg_error(getObject()) << errors[i];
    }

    return res;
}

SOFA_DECL_CLASS(NodeElement)

helper::Creator<BaseElement::NodeFactory, NodeElement> NodeNodeClass("Node");

const char* NodeElement::getClass() const
{
    return NodeNodeClass.c_str();
}

} // namespace xml

} // namespace simulation

} // namespace sofa

