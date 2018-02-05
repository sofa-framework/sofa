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
#ifndef SOFA_SIMULATION_COMMON_XML_ELEMENT_H
#define SOFA_SIMULATION_COMMON_XML_ELEMENT_H

#include <vector>
#include <SofaSimulationCommon/xml/BaseElement.h>
#include <sofa/helper/Factory.h>

namespace sofa
{

namespace simulation
{

namespace xml
{

template<class Object>
class Element : public BaseElement
{
private:
    typename Object::SPtr object;
public:
    Element(const std::string& name, const std::string& type, BaseElement* newParent=NULL);

    virtual ~Element();

    Object* getTypedObject();

    virtual void setObject(typename Object::SPtr newObject);

    /// Get the associated object
    virtual core::objectmodel::Base* getObject();

    typedef helper::Factory< std::string, Object, Element<Object>*, typename Object::SPtr > Factory;
};


} // namespace xml

} // namespace simulation

} // namespace sofa

#include "Element.inl"

#endif
