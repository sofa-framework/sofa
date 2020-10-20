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
#ifndef SOFA_DEFAULTTYPE_TYPEINFO_DATATYPEINFO_VECTYPES_H
#define SOFA_DEFAULTTYPE_TYPEINFO_DATATYPEINFO_VECTYPES_H

#include <sofa/defaulttype/typeinfo/DataTypeInfo_FixedArray.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa::defaulttype
{

template<std::size_t N, typename real>
struct DataTypeInfo< sofa::defaulttype::Vec<N,real> > : public FixedArrayTypeInfo<sofa::defaulttype::Vec<N,real> >
{
    static std::string name() { std::ostringstream o; o << "Vec<" << N << "," << DataTypeInfo<real>::name() << ">"; return o.str(); }
};

template<std::size_t N, typename real>
struct DataTypeInfo< sofa::defaulttype::VecNoInit<N,real> > : public FixedArrayTypeInfo<sofa::defaulttype::VecNoInit<N,real> >
{
    static std::string name() { std::ostringstream o; o << "VecNoInit<" << N << "," << DataTypeInfo<real>::name() << ">"; return o.str(); }
};



// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

#define DataTypeInfoName(type,suffix)\
template<std::size_t N>\
struct DataTypeInfo< sofa::defaulttype::Vec<N,type> > : public FixedArrayTypeInfo<sofa::defaulttype::Vec<N,type> >\
{\
    static std::string name() { std::ostringstream o; o << "Vec" << N << suffix; return o.str(); }\
};\
template<std::size_t N>\
struct DataTypeInfo< sofa::defaulttype::VecNoInit<N,type> > : public FixedArrayTypeInfo<sofa::defaulttype::VecNoInit<N,type> >\
{\
    static std::string name() { std::ostringstream o; o << "VecNoInit" << N << suffix; return o.str(); }\
};

DataTypeInfoName( float, "f" )
DataTypeInfoName( double, "d" )
DataTypeInfoName( int, "i" )
DataTypeInfoName( unsigned, "u" )

#undef DataTypeInfoName

/// \endcond

} /// namespace sofa::defaulttype

#endif /// ENDIT SOFA_DEFAULTTYPE_TYPEINFO_DATATYPEINFO_VECTYPES_H
