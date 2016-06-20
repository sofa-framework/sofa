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
#ifndef SOFA_SIMULATION_XML_ELEMENTNAMEHELPER
#define SOFA_SIMULATION_XML_ELEMENTNAMEHELPER


#include <map>
#include <string>

namespace sofa
{
namespace simulation
{
namespace xml
{


class ElementNameHelper
{
protected:
    std::map<std::string, int> instanceCounter;
    void registerName(const std::string& name);

public:
    ElementNameHelper();
    ~ElementNameHelper(); //terminal class.

    std::string resolveName(const std::string& type, const std::string& name);
};

}
}
}


#endif // SOFA_SIMULATION_XML_ELEMENTNAMEHELPER
