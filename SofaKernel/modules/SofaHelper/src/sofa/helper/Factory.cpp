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
#define SOFAHELPER_FACTORY_CPP
#include <sofa/helper/Factory.inl>
#include <typeinfo>
#ifdef __GNUC__
#include <cxxabi.h>
#endif
#include <cstdlib>

namespace sofa::helper
{

/// Decode the type's name to a more readable form if possible
SOFA_HELPER_API std::string gettypename(const std::type_info& t)
{
    std::string name;
#ifdef __GNUC__
    char* realname = nullptr;
    int status;
    realname = abi::__cxa_demangle(t.name(), 0, 0, &status);
    if (realname!=nullptr)
    {
        int length = 0;
        while(realname[length] != '\0')
        {
            length++;
        }
        name.resize(length);
        for(int i=0; i<(int)length; i++)
            name[i] = realname[i];
        free(realname);
    }
#else
    name = t.name();
#endif
    // Remove namespaces
    for(;;)
    {
        std::string::size_type pos = name.find("::");
        if (pos == std::string::npos) break;
        std::string::size_type first = name.find_last_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",pos-1);
        if (first == std::string::npos) first = 0;
        else first++;
        name.erase(first,pos-first+2);
    }
    //Remove "class "
    for(;;)
    {
        std::string::size_type pos = name.find("class ");
        if (pos == std::string::npos) break;
        name.erase(pos,6);
    }
	//Remove "struct "
    for(;;)
    {
        std::string::size_type pos = name.find("struct ");
        if (pos == std::string::npos) break;
        name.erase(pos,7);
    }
    return name;
}

SOFA_HELPER_API std::string& getFactoryLog()
{
    static std::string s;
    return s;
}

/// Print factory log
SOFA_HELPER_API void printFactoryLog(std::ostream& out)
{
    out << getFactoryLog();
}

//explicit instantiation for std::string
template SOFA_HELPER_API void logFactoryRegister<std::string>(const std::string& baseclass, const std::string& classname, std::string key, bool multi);

} // namespace sofa::helper

