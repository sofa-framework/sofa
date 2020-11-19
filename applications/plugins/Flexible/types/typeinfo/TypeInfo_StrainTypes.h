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
#include <sofa/defaulttype/typeinfo/DataTypeInfo.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>
#include <Flexible/types/StrainTypes.h>

namespace sofa::defaulttype
{

template<> struct DataTypeInfo< E331fTypes::Coord > : public FixedArrayTypeInfo< E331fTypes::Coord, E331fTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E331<" << DataTypeInfo<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E331dTypes::Coord > : public FixedArrayTypeInfo< E331dTypes::Coord, E331dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E331<" << DataTypeInfo<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E321fTypes::Coord > : public FixedArrayTypeInfo< E321fTypes::Coord, E321fTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E321<" << DataTypeInfo<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E321dTypes::Coord > : public FixedArrayTypeInfo< E321dTypes::Coord, E321dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E321<" << DataTypeInfo<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E311fTypes::Coord > : public FixedArrayTypeInfo< E311fTypes::Coord, E311fTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E311<" << DataTypeInfo<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E311dTypes::Coord > : public FixedArrayTypeInfo< E311dTypes::Coord, E311dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E311<" << DataTypeInfo<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E332fTypes::Coord > : public FixedArrayTypeInfo< E332fTypes::Coord, E332fTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E332<" << DataTypeInfo<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E332dTypes::Coord > : public FixedArrayTypeInfo< E332dTypes::Coord, E332dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E332<" << DataTypeInfo<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E333fTypes::Coord > : public FixedArrayTypeInfo< E333fTypes::Coord, E333fTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E333<" << DataTypeInfo<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E333dTypes::Coord > : public FixedArrayTypeInfo< E333dTypes::Coord, E333dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E333<" << DataTypeInfo<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E221fTypes::Coord > : public FixedArrayTypeInfo< E221fTypes::Coord, E221fTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E221<" << DataTypeInfo<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E221dTypes::Coord > : public FixedArrayTypeInfo< E221dTypes::Coord, E221dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "E221<" << DataTypeInfo<double>::name() << ">"; return o.str(); } };

template<> struct DataTypeInfo< I331fTypes::Coord > : public FixedArrayTypeInfo< I331fTypes::Coord, I331fTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "I331<" << DataTypeInfo<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< I331dTypes::Coord > : public FixedArrayTypeInfo< I331dTypes::Coord, I331dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "I331<" << DataTypeInfo<double>::name() << ">"; return o.str(); } };

template<> struct DataTypeInfo< U331fTypes::Coord > : public FixedArrayTypeInfo< U331fTypes::Coord, U331fTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "U331<" << DataTypeInfo<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< U331dTypes::Coord > : public FixedArrayTypeInfo< U331dTypes::Coord, U331dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "U331<" << DataTypeInfo<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< U321fTypes::Coord > : public FixedArrayTypeInfo< U321fTypes::Coord, U321fTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "U321<" << DataTypeInfo<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< U321dTypes::Coord > : public FixedArrayTypeInfo< U321dTypes::Coord, U321dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "U321<" << DataTypeInfo<double>::name() << ">"; return o.str(); } };

}
