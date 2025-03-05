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
#include <sofa/simulation/common/xml/NodeElement.h>
//#include <sofa/simulation/common/xml/ObjectElement.h>
#include <sofa/simulation/common/xml/Element.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::simulation::xml
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
    if (newParent != nullptr && dynamic_cast<NodeElement*>(newParent)==nullptr)
        return false;
    else
        return Element<core::objectmodel::BaseNode>::setParent(newParent);
}

bool NodeElement::initNode()
{
    const core::objectmodel::BaseNode::SPtr obj = Factory::CreateObject(this->getType(), this);
    if (obj != nullptr)
    {
        setObject(obj);
        core::objectmodel::BaseNode* baseNode;
        if (getTypedObject()!=nullptr && getParentElement()!=nullptr && (baseNode = getParentElement()->getObject()->toBaseNode()))
        {
            getTypedObject()->setInstanciationSourceFilePos(getSrcLine());
            getTypedObject()->setInstanciationSourceFileName(getSrcFile());
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
    const bool res = Element<core::objectmodel::BaseNode>::init();

    /// send the errors created by the object in this node in the node's log
    for (unsigned int i=0; i<errors.size(); ++i)
    {
        msg_error(getObject()) << errors[i];
    }

    return res;
}

helper::Creator<BaseElement::NodeFactory, NodeElement> NodeNodeClass("Node");

const char* NodeElement::getClass() const
{
    return NodeNodeClass.getKey().c_str();
}

} // namespace sofa::simulation::xml
