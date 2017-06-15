/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
SOFA_DECL_CLASS(HexahedronSetGeometryAlgorithms)
int HexahedronSetGeometryAlgorithmsClass = core::RegisterObject("Hexahedron set geometry algorithms")
#ifdef SOFA_FLOAT
        .add< HexahedronSetGeometryAlgorithms<Vec3fTypes> >(true) // default template
#else
        .add< HexahedronSetGeometryAlgorithms<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< HexahedronSetGeometryAlgorithms<Vec3fTypes> >() // default template
#endif
#endif
#ifndef SOFA_FLOAT
        .add< HexahedronSetGeometryAlgorithms<Vec2dTypes> >()
        .add< HexahedronSetGeometryAlgorithms<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< HexahedronSetGeometryAlgorithms<Vec2fTypes> >()
        .add< HexahedronSetGeometryAlgorithms<Vec1fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_TOPOLOGY_API HexahedronSetGeometryAlgorithms<Vec3dTypes>;
template class SOFA_BASE_TOPOLOGY_API HexahedronSetGeometryAlgorithms<Vec2dTypes>;
template class SOFA_BASE_TOPOLOGY_API HexahedronSetGeometryAlgorithms<Vec1dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_BASE_TOPOLOGY_API HexahedronSetGeometryAlgorithms<Vec3fTypes>;
template class SOFA_BASE_TOPOLOGY_API HexahedronSetGeometryAlgorithms<Vec2fTypes>;
template class SOFA_BASE_TOPOLOGY_API HexahedronSetGeometryAlgorithms<Vec1fTypes>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

