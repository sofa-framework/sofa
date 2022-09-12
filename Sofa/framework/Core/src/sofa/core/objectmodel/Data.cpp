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
#define SOFA_CORE_OBJECTMODEL_DATA_CPP

#include <sofa/core/objectmodel/Data.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/// Specialization for reading strings
template<>
bool SOFA_CORE_API Data<std::string>::read( const std::string& str )
{
    setValue(str);
    return true;
}

template class SOFA_CORE_API Data< std::string >;
template class SOFA_CORE_API Data< sofa::type::vector<std::string> >;
template class SOFA_CORE_API Data< sofa::type::vector<Index> >;

} // objectmodel

} // core

} // sofa


