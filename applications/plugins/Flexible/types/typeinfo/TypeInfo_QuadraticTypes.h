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
#include <Flexible/types/QuadraticTypes.h>

namespace sofa::defaulttype
{

template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3dTypes::Coord > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3dTypes::Coord, sofa::defaulttype::Quadratic3dTypes::Coord::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticCoord<" << sofa::defaulttype::Quadratic3dTypes::Coord::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3dTypes::Real>::name() << ">"; return o.str(); }
};

template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3dTypes::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3dTypes::Deriv, sofa::defaulttype::Quadratic3dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticDeriv<" << sofa::defaulttype::Quadratic3dTypes::Deriv::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3dTypes::Real>::name() << ">"; return o.str(); }
};

template<> struct DataTypeName< defaulttype::Quadratic3dTypes::Coord > { static const char* name() { return "Quadratic3dTypes::Coord"; } };

template<> struct DataTypeName< defaulttype::Quadratic3dMass > { static const char* name() { return "Quadratic3dMass"; } };

}

