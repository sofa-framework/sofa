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
#include <sofa/core/topology/typeinfo/TypeInfo_Topology.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Vector.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>

namespace sofa::defaulttype
{
using sofa::helper::vector;
using namespace sofa::core::topology;

REGISTER_TYPE_INFO_CREATOR(sofa::core::topology::Topology::Point);
REGISTER_TYPE_INFO_CREATOR(sofa::core::topology::Topology::Edge);
REGISTER_TYPE_INFO_CREATOR(sofa::core::topology::Topology::Triangle);
REGISTER_TYPE_INFO_CREATOR(sofa::core::topology::Topology::Quad);
REGISTER_TYPE_INFO_CREATOR(sofa::core::topology::Topology::Tetrahedron);
REGISTER_TYPE_INFO_CREATOR(sofa::core::topology::Topology::Pyramid);
REGISTER_TYPE_INFO_CREATOR(sofa::core::topology::Topology::Pentahedron);
REGISTER_TYPE_INFO_CREATOR(sofa::core::topology::Topology::Hexahedron);

REGISTER_TYPE_INFO_CREATOR(vector<Topology::Point>);
REGISTER_TYPE_INFO_CREATOR(vector<Topology::Edge>);
REGISTER_TYPE_INFO_CREATOR(vector<Topology::Triangle>);
REGISTER_TYPE_INFO_CREATOR(vector<Topology::Quad>);
REGISTER_TYPE_INFO_CREATOR(vector<Topology::Tetrahedron>);
REGISTER_TYPE_INFO_CREATOR(vector<Topology::Pyramid>);
REGISTER_TYPE_INFO_CREATOR(vector<Topology::Pentahedron>);
REGISTER_TYPE_INFO_CREATOR(vector<Topology::Hexahedron>);

REGISTER_TYPE_INFO_CREATOR(vector<vector<Topology::Point>>);
REGISTER_TYPE_INFO_CREATOR(vector<vector<Topology::Edge>>);
REGISTER_TYPE_INFO_CREATOR(vector<vector<Topology::Triangle>>);
REGISTER_TYPE_INFO_CREATOR(vector<vector<Topology::Quad>>);
REGISTER_TYPE_INFO_CREATOR(vector<vector<Topology::Tetrahedron>>);
REGISTER_TYPE_INFO_CREATOR(vector<vector<Topology::Pyramid>>);
REGISTER_TYPE_INFO_CREATOR(vector<vector<Topology::Pentahedron>>);
REGISTER_TYPE_INFO_CREATOR(vector<vector<Topology::Hexahedron>>);

} /// namespace sofa::defaulttype

