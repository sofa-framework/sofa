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
#define SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDGEOMETRYALGORITHMS_CPP

#include <sofa/component/topology/container/dynamic/DynamicSparseGridGeometryAlgorithms.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::container::dynamic
{

using namespace sofa::defaulttype;

void registerDynamicSparseGridGeometryAlgorithms(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Dynamic sparse grid geometry algorithms.")
        .add< DynamicSparseGridGeometryAlgorithms<Vec3Types> >(true) // default template
        .add< DynamicSparseGridGeometryAlgorithms<Vec2Types> >());
}

template <>
int DynamicSparseGridGeometryAlgorithms<Vec2Types>::findNearestElementInRestPos(const Coord& pos, sofa::type::Vec3& baryC, Real& distance) const
{
    return HexahedronSetGeometryAlgorithms<Vec2Types>::findNearestElementInRestPos(pos, baryC, distance);
}

template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API DynamicSparseGridGeometryAlgorithms<Vec3Types>;
template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API DynamicSparseGridGeometryAlgorithms<Vec2Types>;

} // namespace sofa::component::topology::container::dynamic
