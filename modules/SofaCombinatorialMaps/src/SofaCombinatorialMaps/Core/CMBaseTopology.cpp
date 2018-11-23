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
#include <SofaCombinatorialMaps/Core/CMBaseTopology.h>

namespace sofa
{

namespace core
{

namespace cm_topology
{
// GeometryAlgorithms implementation

void GeometryAlgorithms::init()
{
}

void GeometryAlgorithms::initPointsAdded(const helper::vector< unsigned int >& /*indices*/, const helper::vector< PointAncestorElem >& /*ancestorElems*/
    , const helper::vector< core::VecCoordId >& /*coordVecs*/, const helper::vector< core::VecDerivId >& /*derivVecs */)
{
}

// TopologyAlgorithms implementation

void TopologyAlgorithms::init()
{
    this->getContext()->get(m_topology);
}

void TopologyAlgorithms::addTopologyChange(const TopologyChange *topologyChange)
{
    m_topology->addTopologyChange(topologyChange);
}

// TopologyModifier implementation

void TopologyModifier::init()
{
    this->getContext()->get(m_topology);
}

void TopologyModifier::addTopologyChange(const TopologyChange *topologyChange)
{
    m_topology->addTopologyChange(topologyChange);
}

void TopologyModifier::addStateChange(const TopologyChange *topologyChange)
{
    m_topology->addStateChange(topologyChange);
}

void TopologyModifier::propagateStateChanges() {}
void TopologyModifier::propagateTopologicalChanges() {}
void TopologyModifier::notifyEndingEvent() {}
void TopologyModifier::removeItems(const sofa::helper::vector< unsigned int >& /*items*/) {}


} // namespace cm_topology

} // namespace core

} // namespace sofa

