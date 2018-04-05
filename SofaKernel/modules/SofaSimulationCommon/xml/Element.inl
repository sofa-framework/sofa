/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_COMMON_XML_ELEMENT_INL
#define SOFA_SIMULATION_COMMON_XML_ELEMENT_INL

#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace simulation
{

namespace xml
{


template<class Object>
Element<Object>::Element(const std::string& name, const std::string& type, BaseElement* newParent)
    : BaseElement(name, type, newParent), object(NULL)
{
}

template<class Object>
Element<Object>::~Element()
{
}

template<class Object>
Object* Element<Object>::getTypedObject()
{
    return object.get();
}

template<class Object>
void Element<Object>::setObject(typename Object::SPtr newObject)
{
    object = newObject;
}

/// Get the associated object
template<class Object>
core::objectmodel::Base* Element<Object>::getObject()
{
    return object.get();
}


//template<class Object> class Factory< std::string, Object, Node<Object>* >;


} // namespace xml

} // namespace simulation

} // namespace sofa

#endif
