/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/topology/CMTopologyElementHandler.h>

namespace sofa
{

namespace core
{

namespace cm_topology
{

/// Apply moving elements.
template<class TopologyElementType>
void TopologyElementHandler<TopologyElementType>::ApplyTopologyChange(const EMoved* /*event*/)
{
    std::cerr << "ERROR: MOVED topology event not handled on " << ElementInfo::name()
        << " (it should not even exist!)" << std::endl;
}

/// Apply moving elements on points
template<>
void TopologyElementHandler<topology::MapTopology::Vertex>::ApplyTopologyChange(const EMoved* event)
{
    this->move(event->getIndexArray(), event->ancestorsList, event->baryCoefsList);
}

/// Apply adding function on moved point.
template<>
void TopologyElementHandler<topology::MapTopology::Vertex>::ApplyTopologyChange(const EMoved_Adding* /* event */)
{
    std::cerr << "ERROR: MOVED_ADDING topology event not handled on " << ElementInfo::name()
        << " (it should not even exist!)" << std::endl;
}


/// Apply removing function on moved elements.
template<>
void TopologyElementHandler<topology::MapTopology::Vertex>::ApplyTopologyChange(const EMoved_Removing* /* event */)
{
    std::cerr << "ERROR: MOVED_REMOVING topology event not handled on " << ElementInfo::name()
        << " (it should not even exist!)" << std::endl;
}

template class SOFA_CORE_API TopologyElementHandler<topology::MapTopology::Vertex>;
template class SOFA_CORE_API TopologyElementHandler<topology::MapTopology::Edge>;
template class SOFA_CORE_API TopologyElementHandler<topology::MapTopology::Face>;
template class SOFA_CORE_API TopologyElementHandler<topology::MapTopology::Volume>;

} // namespace cm_topology

} // namespace core

} // namespace sofa
