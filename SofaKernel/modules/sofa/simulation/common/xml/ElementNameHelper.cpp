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
#include <sofa/simulation/common/xml/ElementNameHelper.h>
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
