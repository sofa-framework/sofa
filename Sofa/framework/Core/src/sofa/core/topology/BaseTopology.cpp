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

namespace sofa::core::topology
{

namespace
{
    constexpr sofa::Size getElementTypeIndex(sofa::geometry::ElementType elementType)
    {
        return static_cast<std::underlying_type_t<sofa::geometry::ElementType>>(elementType);
    }
}


// GeometryAlgorithms implementation

void GeometryAlgorithms::doBaseObjectInit()
{
}

void GeometryAlgorithms::initPointsAdded(const type::vector< sofa::Index >& /*indices*/, const type::vector< PointAncestorElem >& /*ancestorElems*/
    , const type::vector< core::VecCoordId >& /*coordVecs*/, const type::vector< core::VecDerivId >& /*derivVecs */)
{
}

// TopologyModifier implementation

void TopologyModifier::doBaseObjectInit()
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
void TopologyModifier::removeItems(const sofa::type::vector< Index >& /*items*/) {}

// TopologyContainer implementation


TopologyContainer::~TopologyContainer()
{
    resetTopologyChangeList();
    resetStateChangeList();
    resetTopologyHandlerList();
}

void TopologyContainer::doBaseObjectInit()
{
    core::topology::BaseMeshTopology::doBaseObjectInit();
    core::topology::BaseTopologyObject::doBaseObjectInit();
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

void TopologyContainer::addTopologyHandler(TopologyHandler *_TopologyHandler, sofa::geometry::ElementType elementType)
{
    m_topologyHandlerListPerElement[getElementTypeIndex(elementType)].push_back(_TopologyHandler);
}

const std::list<TopologyHandler*>& TopologyContainer::getTopologyHandlerList(sofa::geometry::ElementType elementType) const
{
    return m_topologyHandlerListPerElement[getElementTypeIndex(elementType)];
}

bool TopologyContainer::linkTopologyHandlerToData(TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType)
{
    // default implementation dont do anything
    // as it does not hold any data itself
    SOFA_UNUSED(topologyHandler);
    SOFA_UNUSED(elementType);

    return false;
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
    for (auto& topologyHandlerList : m_topologyHandlerListPerElement)
    {
        for (auto it = topologyHandlerList.begin();
            it != topologyHandlerList.end(); ++it)
        {
            *it = nullptr;
        }
        topologyHandlerList.clear();
    }

}

} // namespace sofa::core::topology
