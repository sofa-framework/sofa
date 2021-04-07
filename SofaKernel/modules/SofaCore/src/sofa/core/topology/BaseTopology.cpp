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
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/topology/TopologyHandler.h>

namespace sofa
{

namespace core
{

namespace topology
{
// GeometryAlgorithms implementation

void GeometryAlgorithms::init()
{
}

void GeometryAlgorithms::initPointsAdded(const helper::vector< sofa::Index >& /*indices*/, const helper::vector< PointAncestorElem >& /*ancestorElems*/
    , const helper::vector< core::VecCoordId >& /*coordVecs*/, const helper::vector< core::VecDerivId >& /*derivVecs */)
{
}

// TopologyModifier implementation

void TopologyModifier::init()
{
    this->getContext()->get(m_topologyContainer);
}

void TopologyModifier::addTopologyChange(const TopologyChange *topologyChange)
{
    m_topologyContainer->addTopologyChange(topologyChange);
}

void TopologyModifier::addStateChange(const TopologyChange *topologyChange)
{
    m_topologyContainer->addStateChange(topologyChange);
}

void TopologyModifier::propagateStateChanges() {}
void TopologyModifier::propagateTopologicalChanges() {}
void TopologyModifier::notifyEndingEvent() {}
void TopologyModifier::removeItems(const sofa::helper::vector< Index >& /*items*/) {}

// TopologyContainer implementation


TopologyContainer::~TopologyContainer()
{
    resetTopologyChangeList();
    resetStateChangeList();
    resetTopologyHandlerList();
}

void TopologyContainer::init()
{
    core::topology::BaseMeshTopology::init();
    core::topology::BaseTopologyObject::init();
}


void TopologyContainer::addTopologyChange(const TopologyChange *topologyChange)
{
    std::list<const TopologyChange *>& my_changeList = *(m_changeList.beginEdit());
    my_changeList.push_back(topologyChange);
    m_changeList.endEdit();
}

void TopologyContainer::addStateChange(const TopologyChange *topologyChange)
{
    std::list<const TopologyChange *>& my_stateChangeList = *(m_stateChangeList.beginEdit());
    my_stateChangeList.push_back(topologyChange);
    m_stateChangeList.endEdit();
}

void TopologyContainer::addTopologyHandler(TopologyHandler *_TopologyHandler)
{
    m_TopologyHandlerList.push_back(_TopologyHandler);
    m_TopologyHandlerList.back()->m_changeList.setParent(&this->m_changeList);
    this->updateTopologyHandlerGraph();
}


std::list<const TopologyChange *>::const_iterator TopologyContainer::endChange() const
{
    return (m_changeList.getValue()).end();
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::beginChange() const
{
    return (m_changeList.getValue()).begin();
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::endStateChange() const
{
    return (m_stateChangeList.getValue()).end();
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::beginStateChange() const
{
    return (m_stateChangeList.getValue()).begin();
}

std::list<TopologyHandler *>::const_iterator TopologyContainer::endTopologyHandler() const
{
    return m_TopologyHandlerList.end();
}

std::list<TopologyHandler *>::const_iterator TopologyContainer::beginTopologyHandler() const
{
    return m_TopologyHandlerList.begin();
}

void TopologyContainer::resetTopologyChangeList()
{
    std::list<const TopologyChange *>& my_changeList = *(m_changeList.beginEdit());
    for (std::list<const TopologyChange *>::iterator it=my_changeList.begin();
            it!=my_changeList.end(); ++it)
    {
        delete (*it);
    }

    my_changeList.clear();
    m_changeList.endEdit();
}

void TopologyContainer::resetStateChangeList()
{
    std::list<const TopologyChange *>& my_stateChangeList = *(m_stateChangeList.beginEdit());
    for (std::list<const TopologyChange *>::iterator it=my_stateChangeList.begin();
            it!=my_stateChangeList.end(); ++it)
    {
        delete (*it);
    }

    my_stateChangeList.clear();
    m_stateChangeList.endEdit();
}

void TopologyContainer::resetTopologyHandlerList()
{
    for (std::list<TopologyHandler *>::iterator it=m_TopologyHandlerList.begin();
            it!=m_TopologyHandlerList.end(); ++it)
    {
        //delete (*it);
        *it = nullptr;
    }

    m_TopologyHandlerList.clear();
}


} // namespace topology

} // namespace core

} // namespace sofa

