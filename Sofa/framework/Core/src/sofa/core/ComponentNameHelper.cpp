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
#include <sstream>
#include <sofa/core/ComponentNameHelper.h>
#include <sofa/helper/NameDecoder.h>

namespace sofa::core
{

std::string ComponentNameHelper::resolveName(const std::string& type, const std::string& name, Convention convention)
{
    if (name.empty())
    {
        return resolveName(type, convention);
    }
    return name;
}

std::string ComponentNameHelper::resolveName(const std::string& type, Convention convention)
{
    std::string radix = type;
    if (convention == Convention::xml)
    {
        return radix + std::to_string((m_instanceCounter[radix]++) + 1);
    }

    return radix;
}

}//namespace sofa::core
