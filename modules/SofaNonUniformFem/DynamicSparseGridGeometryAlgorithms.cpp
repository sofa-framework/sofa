/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDGEOMETRYALGORITHMS_CPP
#include <SofaNonUniformFem/DynamicSparseGridGeometryAlgorithms.inl>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
SOFA_DECL_CLASS ( DynamicSparseGridGeometryAlgorithms )
int DynamicSparseGridGeometryAlgorithmsClass = core::RegisterObject ( "Hexahedron set geometry algorithms" )
#ifdef SOFA_FLOAT
        .add< DynamicSparseGridGeometryAlgorithms<Vec3fTypes> > ( true ) // default template
#else
        .add< DynamicSparseGridGeometryAlgorithms<Vec3dTypes> > ( true ) // default template
#ifndef SOFA_DOUBLE
        .add< DynamicSparseGridGeometryAlgorithms<Vec3fTypes> >() // default template
#endif
#endif
#ifndef SOFA_FLOAT
        .add< DynamicSparseGridGeometryAlgorithms<Vec2dTypes> >()
        .add< DynamicSparseGridGeometryAlgorithms<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< DynamicSparseGridGeometryAlgorithms<Vec2fTypes> >()
        .add< DynamicSparseGridGeometryAlgorithms<Vec1fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template <>
int DynamicSparseGridGeometryAlgorithms<Vec2dTypes>::findNearestElementInRestPos(const Coord& pos, sofa::defaulttype::Vector3& baryC, Real& distance) const
{
    return HexahedronSetGeometryAlgorithms<Vec2dTypes>::findNearestElementInRestPos(pos, baryC, distance);
}

template <>
int DynamicSparseGridGeometryAlgorithms<Vec1dTypes>::findNearestElementInRestPos(const Coord& pos, sofa::defaulttype::Vector3& baryC, Real& distance) const
{
    return HexahedronSetGeometryAlgorithms<Vec1dTypes>::findNearestElementInRestPos(pos, baryC, distance);
}

template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<Vec3dTypes>;
template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<Vec2dTypes>;
template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<Vec1dTypes>;
#endif

#ifndef SOFA_DOUBLE
template <>
int DynamicSparseGridGeometryAlgorithms<Vec2fTypes>::findNearestElementInRestPos(const Coord& pos, sofa::defaulttype::Vector3& baryC, Real& distance) const
{
    return HexahedronSetGeometryAlgorithms<Vec2fTypes>::findNearestElementInRestPos( pos, baryC, distance);
}

template <>
int DynamicSparseGridGeometryAlgorithms<Vec1fTypes>::findNearestElementInRestPos(const Coord& pos, sofa::defaulttype::Vector3& baryC, Real& distance) const
{
    return HexahedronSetGeometryAlgorithms<Vec1fTypes>::findNearestElementInRestPos( pos, baryC, distance);
}

template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<Vec3fTypes>;
template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<Vec2fTypes>;
template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<Vec1fTypes>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

