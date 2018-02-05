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

#include <SofaSimulationCommon/xml/DataElement.h>
#include <SofaSimulationCommon/xml/AttributeElement.h>
#include <SofaSimulationCommon/xml/Element.h>
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

