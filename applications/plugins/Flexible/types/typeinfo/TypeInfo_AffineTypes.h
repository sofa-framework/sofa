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
#include <Flexible/types/AffineTypes.h>

namespace sofa::defaulttype
{

// Specialization of the defaulttype::DataTypeInfo type traits template
template<> struct DataTypeInfo< sofa::defaulttype::Affine3dTypes::Coord > : public FixedArrayTypeInfo< sofa::defaulttype::Affine3dTypes::Coord, sofa::defaulttype::Affine3dTypes::Coord::total_size >
{
    static std::string name() { std::ostringstream o; o << "AffineCoord<" << sofa::defaulttype::Affine3dTypes::Coord::total_size << "," << DataTypeName<sofa::defaulttype::Affine3dTypes::Real>::name() << ">"; return o.str(); }
};

template<> struct DataTypeInfo< sofa::defaulttype::Affine3dTypes::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::Affine3dTypes::Deriv, sofa::defaulttype::Affine3dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "AffineDeriv<" << sofa::defaulttype::Affine3dTypes::Deriv::total_size << "," << DataTypeName<sofa::defaulttype::Affine3dTypes::Real>::name() << ">"; return o.str(); }
};

template<> struct DataTypeName< defaulttype::Affine3dTypes::Coord >
{
    static std::string name() { return "Affine3dTypes::Coord"; }
};

template<> struct DataTypeName< defaulttype::Affine3dMass >
{
    static std::string name() { return "Affine3dMass"; }
};

template<> struct DataTypeName< defaulttype::Affine3fMass >
{
    static std::string name() { return "Affine3fMass"; }
};

}
