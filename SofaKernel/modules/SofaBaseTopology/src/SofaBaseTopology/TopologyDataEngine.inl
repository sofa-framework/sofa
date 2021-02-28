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
        sofa::core::topology::BaseMeshTopology *_topology,
        sofa::core::topology::TopologyHandler *_topoHandler) :
    m_topologyData(_topologicalData),
    m_topology(nullptr),
    m_topoHandler(_topoHandler),
    m_pointsLinked(false), m_edgesLinked(false), m_trianglesLinked(false),
    m_quadsLinked(false), m_tetrahedraLinked(false), m_hexahedraLinked(false)
{
    m_topology =  dynamic_cast<sofa::core::topology::TopologyContainer*>(_topology);

    if (m_topology == nullptr)
        msg_error() << "Topology is not dynamic";

    if (m_topoHandler == nullptr)
        msg_error() << "Topology Handler not available";
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
    std::string msg = this->name.getValue() + " - doUpdate: Nbr changes: " + std::to_string(m_changeList.getValue().size());
    sofa::helper::AdvancedTimer::stepBegin(msg.c_str());
    this->ApplyTopologyChanges();
    sofa::helper::AdvancedTimer::stepEnd(msg.c_str());
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::registerTopology(sofa::core::topology::BaseMeshTopology *_topology)
{
    m_topology =  dynamic_cast<sofa::core::topology::TopologyContainer*>(_topology);

    if (m_topology == nullptr)
    {
        msg_error() <<"Topology: " << _topology->getName() << " is not dynamic, topology engine on Data '" << m_data_name << "' won't be registered.";
        return;
    }
    else
        m_topology->addTopologyEngine(this);
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::registerTopology()
{
    if (m_topology == nullptr)
    {
        msg_error() << "Current topology is not dynamic, topology engine on Data '" << m_data_name << "' won't be registered.";
        return;
    }
    else
        m_topology->addTopologyEngine(this);
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChanges()
{
    // Rentre ici la premiere fois aussi....
    if(m_topoHandler)
    {
        m_topoHandler->ApplyTopologyChanges(m_changeList.getValue(), m_topology->getNbPoints());

        m_changeList.endEdit();
    }
}


/// Function to link DataEngine with Data array from topology
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::linkToPointDataArray()
{
    if (m_pointsLinked) // avoid second registration
        return;

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
    this->swap(event->index[0], event->index[1]);
}
/// Apply adding elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChange(const EAdded* event)
{
    //this->add(event->getNbAddedElements(), event->getElementArray(),
    //    event->ancestorsList, event->coefs);
    this->add(event->getIndexArray(), event->getElementArray(),
        event->ancestorsList, event->coefs, event->ancestorElems);
}

/// Apply removing elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChange(const ERemoved* event)
{
    this->remove(event->getArray());
}

/// Apply renumbering on elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChange(const ERenumbering* event)
{
    this->renumber(event->getIndexArray());
}

/// Apply moving elements.
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine<TopologyElementType,  VecT>::ApplyTopologyChange(const EMoved* /*event*/)
{
    msg_warning("TopologyElementHandler") << "MOVED topology event not handled on " << ElementInfo::name()
        << " (it should not even exist!)";
}



///////////////////// Private functions on TopologyDataEngine changes /////////////////////////////
template <typename TopologyElementType, typename VecT>
void TopologyDataEngine <TopologyElementType, VecT>::swap(Index i1, Index i2)
{
    container_type& data = *(m_topologyData->beginEdit());
    value_type tmp = data[i1];
    data[i1] = data[i2];
    data[i2] = tmp;
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine <TopologyElementType, VecT>::add(const sofa::helper::vector<Index>& index,
    const sofa::helper::vector< TopologyElementType >& elems,
    const sofa::helper::vector<sofa::helper::vector<Index> >& ancestors,
    const sofa::helper::vector<sofa::helper::vector<double> >& coefs,
    const sofa::helper::vector< AncestorElem >& ancestorElems)
{
    std::size_t nbElements = index.size();
    if (nbElements == 0) return;
    // Using default values
    container_type& data = *(m_topologyData->beginEdit());
    std::size_t i0 = data.size();
    if (i0 != index[0])
    {
        msg_error(this->m_topologyData->getOwner()) << "TopologyDataEngine SIZE MISMATCH in Data "
            << this->m_topologyData->getName() << ": " << nbElements << " "
            << core::topology::TopologyElementInfo<TopologyElementType>::name()
            << " ADDED starting from index " << index[0]
            << " while vector size is " << i0;
        i0 = index[0];
    }
    data.resize(i0 + nbElements);

    const sofa::helper::vector< Index > empty_vecint;
    const sofa::helper::vector< double > empty_vecdouble;

    for (Index i = 0; i < nbElements; ++i)
    {
        value_type& t = data[i0 + i];
        this->applyCreateFunction(Index(i0 + i), t, elems[i],
            (ancestors.empty() || coefs.empty()) ? empty_vecint : ancestors[i],
            (ancestors.empty() || coefs.empty()) ? empty_vecdouble : coefs[i],
            (ancestorElems.empty()) ? nullptr : &ancestorElems[i]);
    }
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine <TopologyElementType, VecT>::move(const sofa::helper::vector<Index>& indexList,
    const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
    const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    container_type& data = *(m_topologyData->beginEdit());

    for (std::size_t i = 0; i < indexList.size(); i++)
    {
        this->applyDestroyFunction(indexList[i], data[indexList[i]]);
        this->applyCreateFunction(indexList[i], data[indexList[i]], ancestors[i], coefs[i]);
    }

    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine <TopologyElementType, VecT>::remove(const sofa::helper::vector<Index>& index)
{

    container_type& data = *(m_topologyData->beginEdit());
    if (data.size() > 0)
    {
        Index last = Index(data.size() - 1);

        for (std::size_t i = 0; i < index.size(); ++i)
        {
            this->applyDestroyFunction(index[i], data[index[i]]);
            this->swap(index[i], last);
            --last;
        }

        data.resize(data.size() - index.size());
    }
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine <TopologyElementType, VecT>::renumber(const sofa::helper::vector<Index>& index)
{
    container_type& data = *(m_topologyData->beginEdit());

    container_type copy = m_topologyData->getValue(); // not very efficient memory-wise, but I can see no better solution...
    for (std::size_t i = 0; i < index.size(); ++i)
        data[i] = copy[index[i]];

    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataEngine <TopologyElementType, VecT>::addOnMovedPosition(const sofa::helper::vector<Index>& indexList,
    const sofa::helper::vector<TopologyElementType>& elems)
{
    container_type& data = *(m_topologyData->beginEdit());

    // Recompute data
    sofa::helper::vector< Index > ancestors;
    sofa::helper::vector< double >  coefs;
    coefs.push_back(1.0);
    ancestors.resize(1);

    for (std::size_t i = 0; i < indexList.size(); i++)
    {
        ancestors[0] = indexList[i];
        this->applyCreateFunction(indexList[i], data[indexList[i]], elems[i], ancestors, coefs);
    }
    m_topologyData->endEdit();
}



template <typename TopologyElementType, typename VecT>
void TopologyDataEngine <TopologyElementType, VecT>::removeOnMovedPosition(const sofa::helper::vector<Index>& indices)
{
    container_type& data = *(m_topologyData->beginEdit());

    for (std::size_t i = 0; i < indices.size(); i++)
        this->applyDestroyFunction(indices[i], data[indices[i]]);

    m_topologyData->endEdit();

    // TODO check why this call.
    //this->remove( indices );
}


} //namespace sofa::component::topology
