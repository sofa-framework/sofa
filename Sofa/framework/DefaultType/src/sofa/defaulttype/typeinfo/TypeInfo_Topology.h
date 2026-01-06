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

#include <sofa/geometry/ElementInfo.h>

namespace sofa::defaulttype
{

template<>
struct DataTypeInfo< sofa::topology::Edge > : public FixedArrayTypeInfo< sofa::topology::Edge >
{
    static std::string name() { return sofa::geometry::ElementInfo<sofa::geometry::Edge>::name(); }
};

template<>
struct DataTypeInfo< sofa::topology::Triangle > : public FixedArrayTypeInfo<sofa::topology::Triangle >
{
    static std::string name() { return sofa::geometry::ElementInfo<sofa::geometry::Triangle>::name(); }
};

template<>
struct DataTypeInfo< sofa::topology::Quad > : public FixedArrayTypeInfo<sofa::topology::Element<sofa::geometry::Quad> >
{
    static std::string name() { return sofa::geometry::ElementInfo<sofa::geometry::Quad>::name(); }
};

template<>
struct DataTypeInfo< sofa::topology::Tetrahedron > : public FixedArrayTypeInfo<sofa::topology::Tetrahedron>
{
    static std::string name() { return sofa::geometry::ElementInfo<sofa::geometry::Tetrahedron>::name(); }
};

template<>
struct DataTypeInfo< sofa::topology::Pyramid > : public FixedArrayTypeInfo<sofa::topology::Pyramid >
{
    static std::string name() { return sofa::geometry::ElementInfo<sofa::geometry::Pyramid>::name(); }
};

template<>
struct DataTypeInfo< sofa::topology::Prism > : public FixedArrayTypeInfo<sofa::topology::Prism>
{
    static std::string name() { return sofa::geometry::ElementInfo<sofa::geometry::Prism>::name(); }
};

template<>
struct DataTypeInfo< sofa::topology::Hexahedron > : public FixedArrayTypeInfo<sofa::topology::Hexahedron>
{
    static std::string name() { return sofa::geometry::ElementInfo<sofa::geometry::Hexahedron>::name(); }
};

} // namespace sofa::defaulttype
