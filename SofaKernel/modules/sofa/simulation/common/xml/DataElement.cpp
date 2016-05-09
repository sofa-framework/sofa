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

#include <sofa/simulation/common/xml/DataElement.h>
#include <sofa/simulation/common/xml/AttributeElement.h>
#include <sofa/simulation/common/xml/Element.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace simulation
{

namespace xml
{

using namespace sofa::defaulttype;
using helper::Creator;

//template class Factory< std::string, objectmodel::BaseObject, Node<objectmodel::BaseObject*>* >;

DataElement::DataElement(const std::string& name, const std::string& type, BaseElement* parent)
    : Element<core::objectmodel::BaseObject>(name, type, parent)
{
}

DataElement::~DataElement()
{
}

bool DataElement::initNode()
{
    AttributeElement *p = dynamic_cast< AttributeElement *>( getParentElement());
    std::string info;
    info = getAttribute( "value", "");
    p->setValue(info);
    return true;
}

SOFA_DECL_CLASS(Data)

Creator<BaseElement::NodeFactory, DataElement> DataNodeClass("Data");

const char* DataElement::getClass() const
{
    return DataNodeClass.c_str();
}

} // namespace xml

} // namespace simulation

} // namespace sofa

