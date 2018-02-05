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
#include <SofaSimulationCommon/xml/AttributeElement.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace simulation
{

namespace xml
{

using namespace sofa::defaulttype;
using helper::Creator;

AttributeElement::AttributeElement(const std::string& name, const std::string& type, BaseElement* parent)
    : Element<core::objectmodel::BaseObject>(name, type, parent)
{
}

AttributeElement::~AttributeElement()
{
}

bool AttributeElement::init()
{
    int i=0;
    for (child_iterator<> it = begin(); it != end(); ++it)
    {
        i++;
        it->initNode();
    }
    return initNode();
}

bool AttributeElement::initNode()
{
    std::string name = getAttribute( "type", "");

    if (this->replaceAttribute.find(name) != this->replaceAttribute.end())
    {
        value=replaceAttribute[name];
    }
    getParentElement()->setAttribute(name, value.c_str());
    return true;
}

SOFA_DECL_CLASS(Attribute)

Creator<BaseElement::NodeFactory, AttributeElement> AttributeNodeClass("Attribute");

const char* AttributeElement::getClass() const
{
    return AttributeNodeClass.c_str();
}

} // namespace xml

} // namespace simulation

} // namespace sofa

