/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_SIMULATION_COMMON_XML_ATTRIBUTEELEMENT_H
#define SOFA_SIMULATION_COMMON_XML_ATTRIBUTEELEMENT_H

#include <SofaSimulationCommon/xml/Element.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace simulation
{

namespace xml
{

class SOFA_SIMULATION_COMMON_API AttributeElement : public Element<core::objectmodel::BaseObject>
{
public:
    AttributeElement(const std::string& name, const std::string& type, BaseElement* parent=NULL);

    virtual ~AttributeElement();

    virtual bool init();

    virtual bool initNode();

    virtual const char* getClass() const;

    void setValue(const std::string _value) {value=_value;}
    std::string getValue() {return value;}
private:
    std::string value;
};

} // namespace xml

} // namespace simulation

} // namespace sofa

#endif
