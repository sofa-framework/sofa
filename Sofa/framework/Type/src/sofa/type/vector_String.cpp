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
#define SOFA_HELPER_VECTOR_STRING_DEFINITION
#include <sofa/type/vector_String.h>
#include <sofa/type/vector_T.inl>

#include <iostream>
#include <sstream>


/// All integral types are considered as extern templates.
namespace sofa::type
{

/// Output stream
/// Specialization for writing vectors of unsigned char
template <>
SOFA_TYPE_API std::ostream& vector<std::string>::write(std::ostream& os) const
{
    std::string separator = "";
    os << "[";
    for (auto& v : (*this))
    {
        os << separator << '"' << v << '"';
        separator = ", ";
    }
    os << "]";
    return os;
}

} // namespace sofa::type

template class SOFA_TYPE_API sofa::type::vector<std::string>;
