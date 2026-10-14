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

#include <type_traits>
#include <sofa/type/MatSym.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>

namespace sofa::defaulttype
{

template<sofa::Size D, typename real>
struct DataTypeInfo< sofa::type::MatSym<D,real> > : public FixedArrayTypeInfo<sofa::type::MatSym<D,real> >
{
    static std::string GetTypeName()
    {
        std::ostringstream o;
        o << "MatSym<" << D << "," << DataTypeInfo<real>::GetTypeName() << ">";
        return o.str();
    }

    static std::string name()
    {
        std::ostringstream o;
        o << "MatSym" << D << DataTypeInfo<real>::name();
        return o.str();
    }
};

} /// namespace sofa::defaulttype
