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

#include <sofa/defaulttype/typeinfo/DataTypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_FixedArray.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa::defaulttype
{

template<std::size_t N, typename real>
struct DataTypeInfo< sofa::defaulttype::Vec<N,real> > : public FixedArrayTypeInfo<sofa::defaulttype::Vec<N,real> >
{
    static std::string GetTypeName() { std::ostringstream o; o << "Vec<" << N << "," << DataTypeInfo<real>::GetTypeName() << ">"; return o.str(); }
    static std::string GetName() { std::ostringstream o; o << "Vec" << N << DataTypeInfo<real>::GetName() ; return o.str(); }
};

template<std::size_t N, typename real>
struct DataTypeInfo< sofa::defaulttype::VecNoInit<N,real> > : public FixedArrayTypeInfo<sofa::defaulttype::VecNoInit<N,real> >
{
    static std::string GetName() { std::ostringstream o; o << "VecNoInit" << N << DataTypeInfo<real>::GetName(); return o.str(); }
    static std::string GetTypeName() { std::ostringstream o; o << "VecNoInit" << N << "" << DataTypeInfo<real>::GetTypeName() ; return o.str(); }
};

} /// namespace sofa::defaulttype

