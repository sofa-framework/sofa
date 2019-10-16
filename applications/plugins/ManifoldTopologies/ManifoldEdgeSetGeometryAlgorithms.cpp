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
#define SOFA_MANIFOLD_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_CPP

#include <ManifoldTopologies/ManifoldEdgeSetGeometryAlgorithms.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
int ManifoldEdgeSetGeometryAlgorithmsClass = core::RegisterObject("ManifoldEdge set geometry algorithms")
        .add< ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Vec3Types> >(true) // default template
        .add< ManifoldEdgeSetGeometryAlgorithms<Vec2Types> >()
        .add< ManifoldEdgeSetGeometryAlgorithms<Vec1Types> >()
        .add< ManifoldEdgeSetGeometryAlgorithms<Rigid3Types> >()
        .add< ManifoldEdgeSetGeometryAlgorithms<Rigid2Types> >()
        ;

template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Vec3Types>;
template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms<Vec2Types>;
template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms<Vec1Types>;
template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms<Rigid3Types>;
template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms<Rigid2Types>;


} // namespace topology

} // namespace component

} // namespace sofa

