/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
int DynamicSparseGridGeometryAlgorithmsClass = core::RegisterObject ( "Hexahedron set geometry algorithms" )
        .add< DynamicSparseGridGeometryAlgorithms<Vec3Types> > ( true ) // default template
        .add< DynamicSparseGridGeometryAlgorithms<Vec2Types> >()
        .add< DynamicSparseGridGeometryAlgorithms<Vec1Types> >()

        ;

template <>
int DynamicSparseGridGeometryAlgorithms<Vec2Types>::findNearestElementInRestPos(const Coord& pos, sofa::defaulttype::Vector3& baryC, Real& distance) const
{
    return HexahedronSetGeometryAlgorithms<Vec2Types>::findNearestElementInRestPos(pos, baryC, distance);
}

template <>
int DynamicSparseGridGeometryAlgorithms<Vec1Types>::findNearestElementInRestPos(const Coord& pos, sofa::defaulttype::Vector3& baryC, Real& distance) const
{
    return HexahedronSetGeometryAlgorithms<Vec1Types>::findNearestElementInRestPos(pos, baryC, distance);
}

template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<Vec3Types>;
template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<Vec2Types>;
template class SOFA_NON_UNIFORM_FEM_API DynamicSparseGridGeometryAlgorithms<Vec1Types>;


} // namespace topology

} // namespace component

} // namespace sofa

