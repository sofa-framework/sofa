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
#include <sofa/defaulttype/typeinfo/DataTypeInfo_FixedArray.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::defaulttype
{

template<std::size_t N, typename real>
struct DataTypeInfo< sofa::defaulttype::RigidDeriv<N,real> > : public FixedArrayTypeInfo< sofa::defaulttype::RigidDeriv<N,real>, sofa::defaulttype::RigidDeriv<N,real>::total_size >
{
    static std::string name() { std::ostringstream o; o << "RigidDeriv<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

template<std::size_t N, typename real>
struct DataTypeInfo< sofa::defaulttype::RigidCoord<N,real> > : public FixedArrayTypeInfo< sofa::defaulttype::RigidCoord<N,real>, sofa::defaulttype::RigidCoord<N,real>::total_size >
{
    static std::string name() { std::ostringstream o; o << "RigidCoord<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES


template<> struct DataTypeName< defaulttype::Rigid2Types::Coord > { static const char* name() { return "Rigid2Types::Coord"; } };
template<> struct DataTypeName< defaulttype::Rigid2Types::Deriv > { static const char* name() { return "Rigid2Types::Deriv"; } };
template<> struct DataTypeName< defaulttype::Rigid3Types::Coord > { static const char* name() { return "Rigid3Types::Coord"; } };
template<> struct DataTypeName< defaulttype::Rigid3Types::Deriv > { static const char* name() { return "Rigid3Types::Deriv"; } };
template<> struct DataTypeName< defaulttype::Rigid2Mass > { static const char* name() { return "Rigid2Mass"; } };
template<> struct DataTypeName< defaulttype::Rigid3Mass > { static const char* name() { return "Rigid3Mass"; } };

} /// namespace sofa::defaulttype

