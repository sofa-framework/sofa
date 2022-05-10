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
#include <sofa/helper/NameDecoder.h>

namespace sofa::helper
{

/// Decode the type's name to a more readable form if possible
SOFA_HELPER_API std::string gettypename(const std::type_info& t)
{
    return NameDecoder::decodeTypeName(t);
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

