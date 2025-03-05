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
#define SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_CPP
#include <sofa/component/topology/container/dynamic/PointSetGeometryAlgorithms.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::topology::container::dynamic
{

using namespace sofa::defaulttype;

void registerPointSetGeometryAlgorithms(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Geometry algorithms dedicated to a point topology.")
        .add< PointSetGeometryAlgorithms<Vec3Types> >(true) // default template
        .add< PointSetGeometryAlgorithms<Vec2Types> >()
        .add< PointSetGeometryAlgorithms<Vec1Types> >());
}

template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API PointSetGeometryAlgorithms<Vec3Types>;
template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API PointSetGeometryAlgorithms<Vec2Types>;
template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API PointSetGeometryAlgorithms<Vec1Types>;
template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API PointSetGeometryAlgorithms<Rigid3Types>;
template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API PointSetGeometryAlgorithms<Rigid2Types>;

} //namespace sofa::component::topology::container::dynamic
