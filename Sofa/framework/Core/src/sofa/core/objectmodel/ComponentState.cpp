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
#include <map>
#include <iostream>
#include <string>
#include <sofa/core/objectmodel/ComponentState.h>

namespace sofa
{
namespace core
{
namespace objectmodel
{


std::ostream& operator<<(std::ostream& o, const ComponentState& s)
{
    static std::map<ComponentState, std::string> s2str= {{ComponentState::Undefined, "Undefined"},
                                                          {ComponentState::Loading, "Loading"},
                                                          {ComponentState::Valid, "Valid"},
                                                          {ComponentState::Dirty, "Dirty"},
                                                          {ComponentState::Busy, "Busy"},
                                                          {ComponentState::Invalid, "Invalid"}};
    return o << s2str[s];
}


std::istream& operator>>(std::istream& i, ComponentState& s)
{
    static std::map<std::string, ComponentState> str2s= {{"Undefined", ComponentState::Undefined},
                                                          {"Loading", ComponentState::Loading},
                                                          {"Valid", ComponentState::Valid},
                                                          {"Dirty", ComponentState::Dirty},
                                                          {"Busy", ComponentState::Busy},
                                                          {"Invalid", ComponentState::Invalid}};
    std::string tmp;
    i >> tmp;
    if(str2s.find(tmp) == str2s.end())
    {
        i.setstate(std::ios::failbit);
        return i;
    }
    s = str2s[tmp];
    return i;
}

}  /// namespace objectmodel
}  /// namespace core
}  /// namespace sofa


