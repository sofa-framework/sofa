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

template <typename TopologyElementType, typename VecT>
TopologyDataHandler< TopologyElementType, VecT>::TopologyDataHandler(t_topologicalData *_topologicalData,
        sofa::core::topology::BaseMeshTopology *_topology, value_type defaultValue)
    : TopologyHandler()
    , m_topologyData(_topologicalData)
{
    SOFA_UNUSED(defaultValue);
    m_topology = dynamic_cast<sofa::core::topology::TopologyContainer*>(_topology);
}


template <typename TopologyElementType, typename VecT>
TopologyDataHandler< TopologyElementType, VecT>::TopologyDataHandler(t_topologicalData* _topologicalData,
    value_type defaultValue)
    : TopologyHandler()
    , m_topologyData(_topologicalData)
{
    SOFA_UNUSED(defaultValue);
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::init()
{
    // Name creation
    if (m_prefix.empty()) m_prefix = "TopologyDataHandler( " + this->m_topologyData->getOwner()->getName() + " )";
    m_data_name = this->m_topologyData->getName();
    this->addOutput(this->m_topologyData);
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::handleTopologyChange()
{
    if (!m_isRegistered || m_topology == nullptr)
        return;

    sofa::core::topology::TopologyHandler::ApplyTopologyChanges(m_topology->m_changeList.getValue(), m_topology->getNbPoints());
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType, VecT>::linkToTopologyDataArray(sofa::geometry::ElementType elementType)
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

        m_isRegistered = true;
    }
    else
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology is not able to link with a valid Data Array, Data '" << m_data_name << "' won't be linked.";
        return;
    }
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType, VecT>::unlinkToElementDataArray(sofa::core::topology::BaseMeshTopology::Point*)
{
    if (m_topology->unlinkTopologyHandlerToData(this, sofa::core::topology::TopologyElementType::POINT))
    {
        m_topology->removeTopologyHandler(this, sofa::core::topology::TopologyElementType::POINT);
        m_pointsLinked = false;
    }
    else
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology is not able to unlink with a Point Data Array, Data '" << m_data_name << "' won't be unlinked.";
        return;
    }
}

template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType, VecT>::unlinkToElementDataArray(sofa::core::topology::BaseMeshTopology::Edge*)
{
    if (m_topology->unlinkTopologyHandlerToData(this, sofa::core::topology::TopologyElementType::EDGE))
    {
        m_topology->removeTopologyHandler(this, sofa::core::topology::TopologyElementType::EDGE);
        m_edgesLinked = false;
    }
    else
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology is not able to unlink with a Edge Data Array, Data '" << m_data_name << "' won't be unlinked.";
        return;
    }
}

template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType, VecT>::unlinkToElementDataArray(sofa::core::topology::BaseMeshTopology::Triangle*)
{
    if (m_topology->unlinkTopologyHandlerToData(this, sofa::core::topology::TopologyElementType::TRIANGLE))
    {
        m_topology->removeTopologyHandler(this, sofa::core::topology::TopologyElementType::TRIANGLE);
        m_trianglesLinked = false;
    }
    else
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology is not able to unlink with a Triangle Data Array, Data '" << m_data_name << "' won't be unlinked.";
        return;
    }
}

template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType, VecT>::unlinkToElementDataArray(sofa::core::topology::BaseMeshTopology::Quad*)
{
    if (m_topology->unlinkTopologyHandlerToData(this, sofa::core::topology::TopologyElementType::QUAD))
    {
        m_topology->removeTopologyHandler(this, sofa::core::topology::TopologyElementType::QUAD);
        m_quadsLinked = false;
    }
    else
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology is not able to unlink with a Quad Data Array, Data '" << m_data_name << "' won't be unlinked.";
        return;
    }
}

template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType, VecT>::unlinkToElementDataArray(sofa::core::topology::BaseMeshTopology::Tetrahedron*)
{
    if (m_topology->unlinkTopologyHandlerToData(this, sofa::core::topology::TopologyElementType::TETRAHEDRON))
    {
        m_topology->removeTopologyHandler(this, sofa::core::topology::TopologyElementType::TETRAHEDRON);
        m_tetrahedraLinked = false;
    }
    else
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology is not able to unlink with a Tetrahedron Data Array, Data '" << m_data_name << "' won't be unlinked.";
        return;
    }
}

template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType, VecT>::unlinkToElementDataArray(sofa::core::topology::BaseMeshTopology::Hexahedron*)
{
    if (m_topology->unlinkTopologyHandlerToData(this, sofa::core::topology::TopologyElementType::HEXAHEDRON))
    {
        m_topology->removeTopologyHandler(this, sofa::core::topology::TopologyElementType::HEXAHEDRON);
        m_hexahedraLinked = false;
    }
    else
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology is not able to unlink with a Hexahedron Data Array, Data '" << m_data_name << "' won't be unlinked.";
        return;
    }
}


/// Apply swap between indices elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::ApplyTopologyChange(const EIndicesSwap* event)
{
    m_topologyData->swap(event->index[0], event->index[1]);
}


/// Apply adding elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::ApplyTopologyChange(const EAdded* event)
{
    m_topologyData->add(event->getIndexArray(), event->getElementArray(),
        event->ancestorsList, event->coefs, event->ancestorElems);
}

/// Apply removing elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::ApplyTopologyChange(const ERemoved* event)
{
    m_topologyData->remove(event->getArray());
}

/// Apply renumbering on elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::ApplyTopologyChange(const ERenumbering* event)
{
    m_topologyData->renumber(event->getIndexArray());
}

/// Apply moving elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::ApplyTopologyChange(const EMoved* /*event*/)
{
    msg_warning(m_topologyData->getOwner()) << "MOVED topology event not handled on " << ElementInfo::name()
        << " (it should not even exist!)";
}



} //namespace sofa::core::topology
