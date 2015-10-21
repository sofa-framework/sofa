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
#include <sofa/simulation/common/xml/NodeElement.h>
//#include <sofa/simulation/common/xml/ObjectElement.h>
#include <sofa/simulation/common/xml/Element.h>

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
        if (getTypedObject()!=NULL && getParentElement()!=NULL && dynamic_cast<core::objectmodel::BaseNode*>(getParentElement()->getObject())!=NULL)
        {
            // 		std::cout << "Adding Child "<<getName()<<" to "<<getParentElement()->getName()<<std::endl;
            dynamic_cast<core::objectmodel::BaseNode*>(getParentElement()->getObject())->addChild(getTypedObject());
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
    //Store the warnings created by the objects
    for (unsigned int i=0; i<errors.size(); ++i)
    {
        const std::string name = getObject()->getClassName() + " \"" + getObject()->getName() + "\"";
        MAINLOGGER( Error, errors[i], name );
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

