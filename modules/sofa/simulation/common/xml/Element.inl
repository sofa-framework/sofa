/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_SIMULATION_COMMON_XML_ELEMENT_INL
#define SOFA_SIMULATION_COMMON_XML_ELEMENT_INL

#include "Element.h"
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
    return object;
}

template<class Object>
void Element<Object>::setObject(Object* newObject)
{
    object = newObject;
}

/// Get the associated object
template<class Object>
core::objectmodel::Base* Element<Object>::getObject()
{
    return object;
}


//template<class Object> class Factory< std::string, Object, Node<Object>* >;


} // namespace xml

} // namespace simulation

} // namespace sofa

#endif
