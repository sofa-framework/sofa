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
#include "ManifoldEdgeSetTopologyAlgorithms.inl"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
int ManifoldEdgeSetTopologyAlgorithmsClass = core::RegisterObject("ManifoldEdge set topology algorithms")
        .add< ManifoldEdgeSetTopologyAlgorithms<Vec3Types> >(true) // default template
        .add< ManifoldEdgeSetTopologyAlgorithms<Vec2Types> >()
        .add< ManifoldEdgeSetTopologyAlgorithms<Vec1Types> >()
        .add< ManifoldEdgeSetTopologyAlgorithms<Rigid3Types> >()
        .add< ManifoldEdgeSetTopologyAlgorithms<Rigid2Types> >()

        ;
template class ManifoldEdgeSetTopologyAlgorithms<Vec3Types>;
template class ManifoldEdgeSetTopologyAlgorithms<Vec2Types>;
template class ManifoldEdgeSetTopologyAlgorithms<Vec1Types>;
template class ManifoldEdgeSetTopologyAlgorithms<Rigid3Types>;
template class ManifoldEdgeSetTopologyAlgorithms<Rigid2Types>;


} // namespace topology

} // namespace component

} // namespace sofa

