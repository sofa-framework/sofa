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
#include <SofaSimulationCommon/xml/ElementNameHelper.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace simulation
{
namespace xml
{


ElementNameHelper::ElementNameHelper()
{

}

ElementNameHelper::~ElementNameHelper()
{

}

std::string ElementNameHelper::resolveName(const std::string& type, const std::string& name)
{
    std::string resolvedName;
    if(name.empty())
    {
        std::string radix = core::ObjectFactory::ShortName(type);
        registerName(radix);
        std::ostringstream oss;
        oss << radix << instanceCounter[radix];
        resolvedName = oss.str();
    }
    else
    {
        resolvedName = name;
    }
    return resolvedName;

}

void ElementNameHelper::registerName(const std::string& name)
{
    if( instanceCounter.find(name) != instanceCounter.end())
    {
        instanceCounter[name]++;
    }
    else
    {
        instanceCounter[name] = 1;
    }
}

}
}
}
