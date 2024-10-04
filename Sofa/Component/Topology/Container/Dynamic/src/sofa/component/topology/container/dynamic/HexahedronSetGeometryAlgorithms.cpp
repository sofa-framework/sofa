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
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETGEOMETRYALGORITHMS_CPP
#include <sofa/component/topology/container/dynamic/HexahedronSetGeometryAlgorithms.inl>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::container::dynamic
{

using namespace sofa::defaulttype;

void registerHexahedronSetGeometryAlgorithms(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Geometry algorithms dedicated to an hexahedral topology.")
        .add< HexahedronSetGeometryAlgorithms<Vec3Types> >(true) // default template
        .add< HexahedronSetGeometryAlgorithms<Vec2Types> >());
}

template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API HexahedronSetGeometryAlgorithms<Vec3Types>;
template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API HexahedronSetGeometryAlgorithms<Vec2Types>;

} //namespace sofa::component::topology::container::dynamic
