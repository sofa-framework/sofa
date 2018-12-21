/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGYALGORITHMS_CPP
#include <SofaBaseTopology/EdgeSetTopologyAlgorithms.inl>

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
int EdgeSetTopologyAlgorithmsClass = core::RegisterObject("Edge set topology algorithms")
        .add< EdgeSetTopologyAlgorithms<Vec3Types> >(true) // default template
        .add< EdgeSetTopologyAlgorithms<Vec2Types> >()
        .add< EdgeSetTopologyAlgorithms<Vec1Types> >()
        .add< EdgeSetTopologyAlgorithms<Rigid3Types> >()
        .add< EdgeSetTopologyAlgorithms<Rigid2Types> >()

        ;
template class SOFA_BASE_TOPOLOGY_API EdgeSetTopologyAlgorithms<Vec3Types>;
template class SOFA_BASE_TOPOLOGY_API EdgeSetTopologyAlgorithms<Vec2Types>;
template class SOFA_BASE_TOPOLOGY_API EdgeSetTopologyAlgorithms<Vec1Types>;
template class SOFA_BASE_TOPOLOGY_API EdgeSetTopologyAlgorithms<Rigid3Types>;
template class SOFA_BASE_TOPOLOGY_API EdgeSetTopologyAlgorithms<Rigid2Types>;


} // namespace topology

} // namespace component

} // namespace sofa

