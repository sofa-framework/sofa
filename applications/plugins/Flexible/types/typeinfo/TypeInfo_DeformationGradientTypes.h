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
#include <Flexible/types/DeformationGradientTypes.h>

namespace sofa::defaulttype
{
template<> struct DataTypeInfo< F331dTypes::Coord > : public FixedArrayTypeInfo< F331dTypes::Coord, F331dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "F331<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F332dTypes::Coord > : public FixedArrayTypeInfo< F332dTypes::Coord, F332dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "F332<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F321dTypes::Coord > : public FixedArrayTypeInfo< F321dTypes::Coord, F321dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "F321<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F311dTypes::Coord > : public FixedArrayTypeInfo< F311dTypes::Coord, F311dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "F311<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F221dTypes::Coord > : public FixedArrayTypeInfo< F221dTypes::Coord, F221dTypes::Coord::total_size > {    static std::string name() { std::ostringstream o; o << "F221<" << DataTypeName<double>::name() << ">"; return o.str(); } };
}
