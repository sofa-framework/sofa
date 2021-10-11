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
#include <SofaBaseTopology/TopologyDataHandler.h>

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa::component::topology
{

template <typename TopologyElementType, typename VecT>
TopologyDataHandler< TopologyElementType, VecT>::TopologyDataHandler(t_topologicalData *_topologicalData,
        sofa::core::topology::BaseMeshTopology *_topology, value_type defaultValue)
    : TopologyHandler()
    , m_topologyData(_topologicalData)
    , m_pointsLinked(false), m_edgesLinked(false), m_trianglesLinked(false)
    , m_quadsLinked(false), m_tetrahedraLinked(false), m_hexahedraLinked(false)
{
    SOFA_UNUSED(defaultValue);
    m_topology = dynamic_cast<sofa::core::topology::TopologyContainer*>(_topology);
}


template <typename TopologyElementType, typename VecT>
TopologyDataHandler< TopologyElementType, VecT>::TopologyDataHandler(t_topologicalData* _topologicalData,
    value_type defaultValue)
    : TopologyHandler()
    , m_topologyData(_topologicalData)
    , m_defaultValue(defaultValue)
    , m_pointsLinked(false), m_edgesLinked(false), m_trianglesLinked(false)
    , m_quadsLinked(false), m_tetrahedraLinked(false), m_hexahedraLinked(false)
{

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
    if (!this->isTopologyDataRegistered() || m_topology == nullptr)
        return;

    sofa::core::topology::TopologyHandler::ApplyTopologyChanges(m_topology->m_changeList.getValue(), m_topology->getNbPoints());
}


/// Function to link DataEngine with Data array from topology
template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::linkToPointDataArray()
{
    if (m_pointsLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Point Data Array.";
        return;
    }

    sofa::component::topology::PointSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::PointSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology can't be cast as PointSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Point Data Array.";
        return;
    }

    _container->d_initPoints.addOutput(this);
    _container->addTopologyHandler(this);
    m_pointsLinked = true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::linkToEdgeDataArray()
{
    if (m_edgesLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Edge Data Array.";
        return;
    }

    sofa::component::topology::EdgeSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::EdgeSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology can't be cast as EdgeSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Edge Data Array.";
        return;
    }

    _container->d_edge.addOutput(this);
    _container->addTopologyHandler(this);
    m_edgesLinked = true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::linkToTriangleDataArray()
{
    if (m_trianglesLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Triangle Data Array.";
        return;
    }

    sofa::component::topology::TriangleSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::TriangleSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology can't be cast as TriangleSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Triangle Data Array.";
        return;
    }

    _container->d_triangle.addOutput(this);
    _container->addTopologyHandler(this);
    m_trianglesLinked = true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::linkToQuadDataArray()
{
    if (m_quadsLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Quad Data Array.";
        return;
    }

    sofa::component::topology::QuadSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::QuadSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology can't be cast as QuadSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Quad Data Array.";
        return;
    }

    _container->d_quad.addOutput(this);
    _container->addTopologyHandler(this);
    m_quadsLinked = true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::linkToTetrahedronDataArray()
{
    if (m_tetrahedraLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Tetrahedron Data Array.";
        return;
    }

    sofa::component::topology::TetrahedronSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::TetrahedronSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology can't be cast as TetrahedronSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Tetrahedron Data Array.";
        return;
    }

    _container->d_tetrahedron.addOutput(this);
    _container->addTopologyHandler(this);
    m_tetrahedraLinked = true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::linkToHexahedronDataArray()
{
    if (m_hexahedraLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Hexahedron Data Array.";
        return;
    }
    
    sofa::component::topology::HexahedronSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::HexahedronSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error(m_topologyData->getOwner()) << "Owner topology can't be cast as HexahedronSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Hexahedron Data Array.";
        return;
    }

    _container->d_hexahedron.addOutput(this);
    _container->addTopologyHandler(this);
    m_hexahedraLinked = true;
}


/// Apply swap between indices elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::ApplyTopologyChange(const EIndicesSwap* event)
{
    m_topologyData->swap(event->index[0], event->index[1]);
}

template<class TopologyElementType, class VecT>
bool TopologyDataHandler<TopologyElementType, VecT>::isTopologyDataRegistered()
{
    return m_topologyData != nullptr;
}

/// Apply adding elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataHandler<TopologyElementType,  VecT>::ApplyTopologyChange(const EAdded* event)
{
    //this->add(event->getNbAddedElements(), event->getElementArray(),
    //    event->ancestorsList, event->coefs);
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



} //namespace sofa::component::topology
