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

#include <sofa/core/topology/Topology.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>

namespace sofa::defaulttype
{

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Edge > : public FixedArrayTypeInfo<sofa::helper::fixed_array<index_type,2> >
{
    static std::string GetName() { return "Edge"; }
    static std::string GetTypeName() { return "Edge"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Triangle > : public FixedArrayTypeInfo<sofa::helper::fixed_array<index_type,3> >
{
    static std::string GetName() { return "Triangle"; }
    static std::string GetTypeName() { return "Triangle"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Quad > : public FixedArrayTypeInfo<sofa::helper::fixed_array<index_type,4> >
{
    static std::string GetName() { return "Quad"; }
    static std::string GetTypeName() { return "Quad"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Tetrahedron > : public FixedArrayTypeInfo<sofa::helper::fixed_array<index_type,4> >
{
    static std::string GetName() { return "Tetrahedron"; }
    static std::string GetTypeName() { return "Tetrahedron"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Pyramid > : public FixedArrayTypeInfo<sofa::helper::fixed_array<index_type,5> >
{
    static std::string GetName() { return "Pyramid"; }
    static std::string GetTypeName() { return "Pyramid"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Pentahedron > : public FixedArrayTypeInfo<sofa::helper::fixed_array<index_type,6> >
{
    static std::string GetName() { return "Pentahedron"; }
    static std::string GetTypeName() { return "Pentahedron"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Hexahedron > : public FixedArrayTypeInfo<sofa::helper::fixed_array<index_type,8> >
{
    static std::string GetName() { return "Hexahedron"; }
    static std::string GetTypeName() { return "Hexahedron"; }
};

} /// namespace sofa::defaulttype
