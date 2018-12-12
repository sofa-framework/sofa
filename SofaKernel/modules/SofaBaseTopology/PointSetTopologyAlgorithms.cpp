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
#include <sofa/core/ObjectFactory.h>
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYALGORITHMS_CPP
#include <SofaBaseTopology/PointSetTopologyAlgorithms.inl>


namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
int PointSetTopologyAlgorithmsClass = core::RegisterObject("Point set topology algorithms")
#ifdef SOFA_FLOAT
        .add< PointSetTopologyAlgorithms<Vec3fTypes> >(true) // default template
#else
        .add< PointSetTopologyAlgorithms<Vec3dTypes> >(true) // default template
#endif
        .add< PointSetTopologyAlgorithms<Vec2Types> >()
        .add< PointSetTopologyAlgorithms<Vec1Types> >()

        ;

template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<Vec3Types>;
template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<Vec2Types>;
template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<Vec1Types>;
template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<Rigid3Types>;
template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<Rigid2Types>;


} // namespace topology

} // namespace component

} // namespace sofa

