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
#pragma once
#include <sofa/simulation/common/xml/Element.h>
#include <sofa/simulation/common/xml/BaseElement.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/simulation/Simulation.h>

namespace sofa::simulation::xml
{

class SOFA_SIMULATION_COMMON_API NodeElement : public Element<sofa::core::objectmodel::BaseNode>
{
public:
    NodeElement(const std::string& name, const std::string& type, BaseElement* parent=nullptr);

    ~NodeElement() override;

    bool setParent(BaseElement* newParent) override;

    bool initNode() override;

    bool init() override;

    virtual const char* getClass() const override;

    typedef Element<sofa::core::objectmodel::BaseNode>::Factory Factory;
};

} // namespace sofa::simulation::xml
