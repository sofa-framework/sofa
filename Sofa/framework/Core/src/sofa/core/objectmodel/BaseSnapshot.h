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
#include <sofa/core/objectmodel/Base.h>

namespace sofa::core::objectmodel
{

class SOFA_CORE_API BaseSnapshot 
{

public:
    virtual void printSnapshot() = 0;
    virtual void exportSnapshot() = 0;
    virtual void importSnapshot() = 0;

    virtual void setName(const std::string& name) = 0;
    virtual std::string getName() const = 0;

    virtual void fillContainer(const std::vector<std::string>& name, int i) = 0;
    virtual std::vector<std::vector<std::string>> getContainer() const = 0;


    BaseSnapshot();
    virtual ~BaseSnapshot() = 0;

public:
    std::string dataName;
    std::string dataValueType;
    std::string valueStr;
    std::vector<std::vector<std::string>> container;
};
} // namespace sofa::core::objectmodel