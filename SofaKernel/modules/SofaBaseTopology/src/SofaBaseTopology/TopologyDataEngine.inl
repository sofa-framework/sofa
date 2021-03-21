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
#include <SofaBaseTopology/TopologyDataEngine.h>

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa::component::topology
{

template <typename TopologyElementType, typename VecT>
TopologyDataEngine< TopologyElementType, VecT>::TopologyDataEngine(t_topologicalData *_topologicalData,
        sofa::core::topology::BaseMeshTopology *_topology, value_type defaultValue)
    : TopologyEngine()
    , m_topologyData(_topologicalData)
    , m_topology(nullptr)
    , m_pointsLinked(false), m_edgesLinked(false), m_trianglesLinked(false)
    , m_quadsLinked(false), m_tetrahedraLinked(false), m_hexahedraLinked(false)
{
    m_topology =  dynamic_cast<sofa::core::topology::TopologyContainer*>(_topology);
}


template <typename TopologyElementType, typename VecT>
TopologyDataEngine< TopologyElementType, VecT>::TopologyDataEngine(t_topologicalData* _topologicalData,
    value_type defaultValue)
    : TopologyEngine()
    , m_topologyData(_topologicalData)
    , m_defaultValue(defaultValue) 
    , m_topology(nullptr)
    , m_pointsLinked(false), m_edgesLinked(false), m_trianglesLinked(false)
    , m_quadsLinked(false), m_tetrahedraLinked(false), m_hexahedraLinked(false)
{

}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::init()
{
    // A pointData is by default child of positionSet Data
    //this->linkToPointDataArray();  // already done while creating engine

    // Name creation
    if (m_prefix.empty()) m_prefix = "TopologyEngine_";
    m_data_name = this->m_topologyData->getName();
    this->addOutput(this->m_topologyData);

    sofa::core::topology::TopologyEngine::init();

    // Register Engine in containter list
    //if (m_topology)
    //   m_topology->addTopologyEngine(this);
    //this->registerTopology(m_topology);
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::reinit()
{
    this->update();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::doUpdate()
{
    if (!this->isTopologyDataRegistered())
        return;

    std::string msg = this->name.getValue() + " - doUpdate: Nbr changes: " + std::to_string(m_changeList.getValue().size());
    sofa::helper::AdvancedTimer::stepBegin(msg.c_str());
    this->ApplyTopologyChanges();
    sofa::helper::AdvancedTimer::stepEnd(msg.c_str());
}


template <typename TopologyElementType, typename VecT>
bool TopologyDataEngine<TopologyElementType,  VecT>::registerTopology(sofa::core::topology::BaseMeshTopology *_topology)
{
    m_topology =  dynamic_cast<sofa::core::topology::TopologyContainer*>(_topology);

    if (m_topology == nullptr)
    {
        msg_info() <<"Topology: " << _topology->getName() << " is not dynamic, topology engine on Data '" << m_data_name << "' won't be registered.";
        return false;
    }
    else
        m_topology->addTopologyEngine(this);

    return true;
}


template <typename TopologyElementType, typename VecT>
bool TopologyDataEngine<TopologyElementType,  VecT>::registerTopology()
{
    if (m_topology == nullptr)
    {
        msg_info() << "Current topology is not dynamic, topology engine on Data '" << m_data_name << "' won't be registered.";
        return false;
    }
    else
        m_topology->addTopologyEngine(this);

    return true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChanges()
{
    if (!this->isTopologyDataRegistered() || m_topology == nullptr)
        return;

    m_topologyData->setDataSetArraySize(m_topology->getNbPoints());

    sofa::core::topology::TopologyEngine::ApplyTopologyChanges(m_changeList.getValue(), m_topology->getNbPoints());
}


/// Function to link DataEngine with Data array from topology
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::linkToPointDataArray()
{
    if (m_pointsLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error() << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Point Data Array.";
        return;
    }

    sofa::component::topology::PointSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::PointSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error() << "Owner topology can't be cast as PointSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Point Data Array.";
        return;
    }

    (_container->getPointDataArray()).addOutput(this);
    m_pointsLinked = true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::linkToEdgeDataArray()
{
    if (m_edgesLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error() << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Edge Data Array.";
        return;
    }

    sofa::component::topology::EdgeSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::EdgeSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error() << "Owner topology can't be cast as EdgeSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Edge Data Array.";
        return;
    }

    (_container->getEdgeDataArray()).addOutput(this);
    m_edgesLinked = true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::linkToTriangleDataArray()
{
    if (m_trianglesLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error() << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Triangle Data Array.";
        return;
    }

    sofa::component::topology::TriangleSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::TriangleSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error() << "Owner topology can't be cast as TriangleSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Triangle Data Array.";
        return;
    }

    (_container->getTriangleDataArray()).addOutput(this);
    m_trianglesLinked = true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::linkToQuadDataArray()
{
    if (m_quadsLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error() << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Quad Data Array.";
        return;
    }

    sofa::component::topology::QuadSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::QuadSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error() << "Owner topology can't be cast as QuadSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Quad Data Array.";
        return;
    }

    (_container->getQuadDataArray()).addOutput(this);
    m_quadsLinked = true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::linkToTetrahedronDataArray()
{
    if (m_tetrahedraLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error() << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Tetrahedron Data Array.";
        return;
    }

    sofa::component::topology::TetrahedronSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::TetrahedronSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error() << "Owner topology can't be cast as TetrahedronSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Tetrahedron Data Array.";
        return;
    }

    (_container->getTetrahedronDataArray()).addOutput(this);
    m_tetrahedraLinked = true;
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::linkToHexahedronDataArray()
{
    if (m_hexahedraLinked) // avoid second registration
        return;

    if (m_topology == nullptr)
    {
        msg_error() << "Owner topology has not been set. Data '" << m_data_name << "' won't be linked to Hexahedron Data Array.";
        return;
    }
    
    sofa::component::topology::HexahedronSetTopologyContainer* _container = dynamic_cast<sofa::component::topology::HexahedronSetTopologyContainer*>(m_topology);
    if (_container == nullptr)
    {
        msg_error() << "Owner topology can't be cast as HexahedronSetTopologyContainer, Data '" << m_data_name << "' won't be linked to Hexahedron Data Array.";
        return;
    }

    (_container->getHexahedronDataArray()).addOutput(this);
    m_hexahedraLinked = true;
}


/// Apply swap between indices elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChange(const EIndicesSwap* event)
{
    m_topologyData->swap(event->index[0], event->index[1]);
}
/// Apply adding elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChange(const EAdded* event)
{
    //this->add(event->getNbAddedElements(), event->getElementArray(),
    //    event->ancestorsList, event->coefs);
    m_topologyData->add(event->getIndexArray(), event->getElementArray(),
        event->ancestorsList, event->coefs, event->ancestorElems);
}

/// Apply removing elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChange(const ERemoved* event)
{
    m_topologyData->remove(event->getArray());
}

/// Apply renumbering on elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChange(const ERenumbering* event)
{
    m_topologyData->renumber(event->getIndexArray());
}

/// Apply moving elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChange(const EMoved* /*event*/)
{
    msg_warning("TopologyDataEngine") << "MOVED topology event not handled on " << ElementInfo::name()
        << " (it should not even exist!)";
}



} //namespace sofa::component::topology
