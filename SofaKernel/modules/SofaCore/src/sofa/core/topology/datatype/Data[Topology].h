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

namespace sofa::core::objectmodel
{

#ifndef SOFA_CORE_TOPOLOGY_DATATYPE_DATATOPOLOGY_DEFINITION
extern template class Data<sofa::core::topology::Topology::Edge>;
extern template class Data<sofa::core::topology::Topology::Triangle>;
extern template class Data<sofa::core::topology::Topology::Quad>;
extern template class Data<sofa::core::topology::Topology::Pyramid>;
extern template class Data<sofa::core::topology::Topology::Tetrahedron>;
extern template class Data<sofa::core::topology::Topology::Pentahedron>;
extern template class Data<sofa::core::topology::Topology::Hexahedron>;

extern template class Data<sofa::helper::vector<sofa::core::topology::Topology::Edge>>;
extern template class Data<sofa::helper::vector<sofa::core::topology::Topology::Triangle>>;
extern template class Data<sofa::helper::vector<sofa::core::topology::Topology::Quad>>;
extern template class Data<sofa::helper::vector<sofa::core::topology::Topology::Pyramid>>;
extern template class Data<sofa::helper::vector<sofa::core::topology::Topology::Tetrahedron>>;
extern template class Data<sofa::helper::vector<sofa::core::topology::Topology::Pentahedron>>;
extern template class Data<sofa::helper::vector<sofa::core::topology::Topology::Hexahedron>>;
#endif ///

}
