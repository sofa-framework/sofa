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
#pragma once

#include <sofa/defaulttype/typeinfo/models/FixedArrayTypeInfo.h>
#include <sofa/helper/fixed_array.h>

namespace sofa::defaulttype
{

template<class T, std::size_t N>
struct DataTypeInfo< sofa::helper::fixed_array<T,N> > : public FixedArrayTypeInfo<sofa::helper::fixed_array<T,N> >
{
    static std::string name() { std::ostringstream o; o << "fixed_array<" << DataTypeInfo<T>::name() << "," << N << ">"; return o.str(); }
    static std::string shortName() { std::ostringstream o; o << "fixed_array<" << DataTypeInfo<T>::shortName() << "," << N << ">"; return o.str(); }
};

} /// namespace sofa::defaulttype

