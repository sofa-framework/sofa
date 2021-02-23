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

#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>
#include <sofa/topology/Topology.h>

namespace sofa::defaulttype
{

template<>
struct DataTypeInfo< sofa::topology::Topology::Edge > : public FixedArrayTypeInfo<sofa::type::stdtype::fixed_array<Index,2> >
{
    static std::string name() { return "Edge"; }
};

template<>
struct DataTypeInfo< sofa::topology::Topology::Triangle > : public FixedArrayTypeInfo<sofa::type::stdtype::fixed_array<Index,3> >
{
    static std::string name() { return "Triangle"; }
};

template<>
struct DataTypeInfo< sofa::topology::Topology::Quad > : public FixedArrayTypeInfo<sofa::type::stdtype::fixed_array<Index,4> >
{
    static std::string name() { return "Quad"; }
};

template<>
struct DataTypeInfo< sofa::topology::Topology::Tetrahedron > : public FixedArrayTypeInfo<sofa::type::stdtype::fixed_array<Index,4> >
{
    static std::string name() { return "Tetrahedron"; }
};

template<>
struct DataTypeInfo< sofa::topology::Topology::Pyramid > : public FixedArrayTypeInfo<sofa::type::stdtype::fixed_array<Index,5> >
{
    static std::string name() { return "Pyramid"; }
};

template<>
struct DataTypeInfo< sofa::topology::Topology::Pentahedron > : public FixedArrayTypeInfo<sofa::type::stdtype::fixed_array<Index,6> >
{
    static std::string name() { return "Pentahedron"; }
};

template<>
struct DataTypeInfo< sofa::topology::Topology::Hexahedron > : public FixedArrayTypeInfo<sofa::type::stdtype::fixed_array<Index,8> >
{
    static std::string name() { return "Hexahedron"; }
};

} // namespace sofa::defaulttype

