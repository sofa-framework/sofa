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

/// Specialization for reading booleans
template<>
void  SOFA_CORE_API Data<bool>::doRead( std::istringstream& str )
{
    bool val;
    int c = str.peek();
    if(c==EOF)
    {
        str.setstate(std::ios::failbit);
        return;
    }

    if (c == 'T' || c == 't')
        val = true;
    else if (c == 'F' || c == 'f')
        val = false;
    else if ((c >= '0' && c <= '9') || c == '-')
    {
        int numericValue;
        str >> numericValue;
        val = (numericValue != 0);
    }
    else{
        str.setstate(std::ios::failbit);
        return;
    }
    setValue(val);
}

template class SOFA_CORE_API Data< std::string >;
template class SOFA_CORE_API Data< sofa::type::vector<std::string> >;
template class SOFA_CORE_API Data< bool >;
template class SOFA_CORE_API Data< sofa::type::vector<Index> >;

} // objectmodel

} // core

} // sofa


