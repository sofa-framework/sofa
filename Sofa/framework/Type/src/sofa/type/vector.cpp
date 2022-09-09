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
#define SOFA_HELPER_VECTOR_DEFINITION
#include <sofa/type/vector.h>

#include <sstream>

namespace sofa::type
{

void SOFA_TYPE_API vector_access_failure(const void* vec, const std::size_t size, const std::size_t i, const std::type_info& type)
{
    std::ostringstream oss;
    oss << "in vector<" << type.name() << "> " << std::hex << vec << std::dec << " size " << size << " : invalid index " << i;
    throw std::logic_error(oss.str());
}


} // namespace sofa::type
