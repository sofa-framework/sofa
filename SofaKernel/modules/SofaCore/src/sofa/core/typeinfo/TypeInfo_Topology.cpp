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
#include <sofa/core/typeinfo/TypeInfo_Topology.h>
#include <sofa/defaulttype/TypeInfoRegistryTools.h>
using sofa::defaulttype::loadCoreContainersInRepositoryForType;

namespace
{

int loadTypeInfoForTopologies()
{
    loadCoreContainersInRepositoryForType<sofa::core::topology::Topology::Point>(sofa_do_tostring(SOFA_TARGET));
    loadCoreContainersInRepositoryForType<sofa::core::topology::Topology::Edge>(sofa_do_tostring(SOFA_TARGET));
    loadCoreContainersInRepositoryForType<sofa::core::topology::Topology::Triangle>(sofa_do_tostring(SOFA_TARGET));
    loadCoreContainersInRepositoryForType<sofa::core::topology::Topology::Quad>(sofa_do_tostring(SOFA_TARGET));
    loadCoreContainersInRepositoryForType<sofa::core::topology::Topology::Tetrahedron>(sofa_do_tostring(SOFA_TARGET));
    loadCoreContainersInRepositoryForType<sofa::core::topology::Topology::Pyramid>(sofa_do_tostring(SOFA_TARGET));
    loadCoreContainersInRepositoryForType<sofa::core::topology::Topology::Pentahedron>(sofa_do_tostring(SOFA_TARGET));
    loadCoreContainersInRepositoryForType<sofa::core::topology::Topology::Hexahedron>(sofa_do_tostring(SOFA_TARGET));
    return 1;
}

static int inited = loadTypeInfoForTopologies();

}
