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
#pragma once
#include <sofa/core/topology/TopologyDataHandler.h>

namespace sofa::core::topology
{

template <typename ElementType, typename VecT>
TopologyDataHandler< ElementType, VecT>::TopologyDataHandler(t_topologicalData *_topologicalData,
        sofa::core::topology::BaseMeshTopology *_topology, value_type defaultValue)
    : TopologyHandler()
    , m_topologyData(_topologicalData)
{
    SOFA_UNUSED(defaultValue);
    m_topology = dynamic_cast<sofa::core::topology::TopologyContainer*>(_topology);
}


template <typename ElementType, typename VecT>
TopologyDataHandler< ElementType, VecT>::TopologyDataHandler(t_topologicalData* _topologicalData,
    value_type defaultValue)
    : TopologyHandler()
    , m_topologyData(_topologicalData)
{
    SOFA_UNUSED(defaultValue);
}


template <typename ElementType, typename VecT>
void TopologyDataHandler<ElementType,  VecT>::init()
{
    // Name creation
    if (m_prefix.empty()) m_prefix = "TopologyDataHandler( " + this->m_topologyData->getOwner()->getName() + " )";
    m_data_name = this->m_topologyData->getName();
    this->addOutput(this->m_topologyData);
}


template <typename ElementType, typename VecT>
void TopologyDataHandler<ElementType,  VecT>::handleTopologyChange()
{
    if (!this->isTopologyHandlerRegistered() || m_topology == nullptr)
        return;

    sofa::core::topology::TopologyHandler::ApplyTopologyChanges(m_topology->m_changeList.getValue(), m_topology->getNbPoints());
}


template <typename ElementType, typename VecT>
void TopologyDataHandler<ElementType, VecT>::linkToTopologyDataArray(sofa::geometry::ElementType elementType)
{
    if (m_topology == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Data Array.";
        return;
    }

    if (m_topology->linkTopologyHandlerToData(this, elementType))
    {
        if (m_topology->addTopologyHandler(this, elementType) == false)
        {
            msg_warning(m_topologyData->getOwner()) << "TopologyHandler linked to Data '" << m_data_name << "' has already been registered.";
        }
        else
        {
            m_registeredElements.insert(elementType);
        }
    }
    else
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology is not able to link with a valid Data Array, Data '" << m_data_name << "' won't be linked.";
        return;
    }
}


template <typename ElementType, typename VecT>
void TopologyDataHandler<ElementType, VecT>::unlinkFromTopologyDataArray(sofa::geometry::ElementType elementType)
{
    const auto it = m_registeredElements.find(elementType);
    if (it == m_registeredElements.end()) // case if this element type has never been registered or topology has already been deleted
        return;

    const bool res = m_topology->unlinkTopologyHandlerToData(this, elementType);
    msg_error_when(!res, m_topologyData->getOwner()) << "Owner topology is not able to unlink with Data Array, Data '" << m_data_name << "' won't be unlinked.";
    
    m_topology->removeTopologyHandler(this, elementType);
    m_registeredElements.erase(it);
}




template <typename ElementType, typename VecT>
void TopologyDataHandler<ElementType, VecT>::unlinkFromAllTopologyDataArray()
{
    if (m_registeredElements.empty()) // Will be false if topology has already been deleted
        return;

    for (auto elementType : m_registeredElements)
    {
        const bool res = m_topology->unlinkTopologyHandlerToData(this, elementType);
        msg_error_when(!res, m_topologyData->getOwner()) << "Owner topology is not able to unlink with Data Array, Data '" << m_data_name << "' won't be unlinked.";

        m_topology->removeTopologyHandler(this, elementType);
    }

    m_registeredElements.clear();
}


/// Apply swap between indices elements.
template <typename ElementType, typename VecT>
void TopologyDataHandler<ElementType,  VecT>::ApplyTopologyChange(const EIndicesSwap* event)
{
    m_topologyData->swap(event->index[0], event->index[1]);
}


/// Apply adding elements.
template <typename ElementType, typename VecT>
void TopologyDataHandler<ElementType,  VecT>::ApplyTopologyChange(const EAdded* event)
{
    m_topologyData->add(event->getIndexArray(), event->getElementArray(),
        event->ancestorsList, event->coefs, event->ancestorElems);
}

/// Apply removing elements.
template <typename ElementType, typename VecT>
void TopologyDataHandler<ElementType,  VecT>::ApplyTopologyChange(const ERemoved* event)
{
    m_topologyData->remove(event->getArray());
}

/// Apply renumbering on elements.
template <typename ElementType, typename VecT>
void TopologyDataHandler<ElementType,  VecT>::ApplyTopologyChange(const ERenumbering* event)
{
    m_topologyData->renumber(event->getIndexArray());
}

/// Apply moving elements.
template <typename ElementType, typename VecT>
void TopologyDataHandler<ElementType,  VecT>::ApplyTopologyChange(const EMoved* /*event*/)
{
    msg_warning(m_topologyData->getOwner()) << "MOVED topology event not handled on " << ElementInfo::name()
        << " (it should not even exist!)";
}



} //namespace sofa::core::topology
