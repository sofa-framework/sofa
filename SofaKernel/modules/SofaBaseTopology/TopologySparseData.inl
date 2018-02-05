/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSPARSEDATA_INL
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSPARSEDATA_INL

#include <SofaBaseTopology/TopologySparseData.h>
#include <SofaBaseTopology/TopologyEngine.inl>
#include <SofaBaseTopology/TopologySparseDataHandler.inl>

namespace sofa
{

namespace component
{

namespace topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TopologyElementType, typename VecT>
TopologySparseDataImpl <TopologyElementType, VecT>::~TopologySparseDataImpl()
{
    if (this->m_topologyHandler)
        delete m_topologyHandler;
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataImpl <TopologyElementType, VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology, sofa::core::topology::TopologyHandler *_topologyHandler)
{
    this->m_topology = _topology;
    if (_topology && dynamic_cast<sofa::core::topology::TopologyContainer*>(_topology))
    {
        this->m_topologicalEngine = sofa::core::objectmodel::New< TopologyEngineImpl<VecT> >((sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>*)this, _topology, _topologyHandler);
        this->m_topologicalEngine->setNamePrefix(std::string(sofa::core::topology::TopologyElementInfo<TopologyElementType>::name()) + std::string("SparseEngine_"));
        if (this->getOwner() && dynamic_cast<sofa::core::objectmodel::BaseObject*>(this->getOwner())) dynamic_cast<sofa::core::objectmodel::BaseObject*>(this->getOwner())->addSlave(this->m_topologicalEngine.get());
        this->m_topologicalEngine->init();
        this->linkToElementDataArray((TopologyElementType*)NULL);
        this->getOwner()->sout<<"TopologySparseDataImpl: " << this->getName() << " initialized with dynamic " << _topology->getClassName() << " Topology." << this->getOwner()->sendl;
    }
    else if (_topology)
    {
        this->getOwner()->sout<<"TopologySparseDataImpl: " << this->getName() << " initialized with static " << _topology->getClassName() << " Topology." << this->getOwner()->sendl;
    }
    else
    {
        this->getOwner()->sout<<"TopologySparseDataImpl: No Topology given to " << this->getName() << " to createTopologicalEngine. Topological changes will be disabled." << this->getOwner()->sendl;
    }
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataImpl <TopologyElementType, VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
    this->m_topologyHandler = new TopologySparseDataHandler<TopologyElementType, VecT>(this);
    createTopologicalEngine(_topology, this->m_topologyHandler);
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataImpl <TopologyElementType, VecT>::registerTopologicalData()
{
    if (this->m_topologicalEngine)
        this->m_topologicalEngine->registerTopology();
    else if (!this->m_topology)
        this->getOwner()->sout<<"TopologySparseDataImpl: " << this->getName() << " has no engine. Topological changes will be disabled. Use createTopologicalEngine method before registerTopologicalData to allow topological changes." << this->getOwner()->sendl;
}

template <typename TopologyElementType, typename VecT>
void TopologySparseDataImpl <TopologyElementType, VecT>::addInputData(sofa::core::objectmodel::BaseData *_data)
{
    if (this->m_topologicalEngine)
        this->m_topologicalEngine->addInput(_data);
    else if (!this->m_topology)
        this->getOwner()->sout<<"Warning: TopologySparseDataImpl: " << this->getName() << " has no engine. Use createTopologicalEngine function before addInputData." << this->getOwner()->sendl;
}



/// Method used to link Data to point Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySparseDataImpl <TopologyElementType, VecT>::linkToPointDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToPointDataArray();
}

/// Method used to link Data to edge Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySparseDataImpl <TopologyElementType, VecT>::linkToEdgeDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToEdgeDataArray();
}

/// Method used to link Data to triangle Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySparseDataImpl <TopologyElementType, VecT>::linkToTriangleDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToTriangleDataArray();
}

/// Method used to link Data to quad Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySparseDataImpl <TopologyElementType, VecT>::linkToQuadDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToQuadDataArray();
}

/// Method used to link Data to tetrahedron Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySparseDataImpl <TopologyElementType, VecT>::linkToTetrahedronDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToTetrahedronDataArray();
}

/// Method used to link Data to hexahedron Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySparseDataImpl <TopologyElementType, VecT>::linkToHexahedronDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToHexahedronDataArray();
}



} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSPARSEDATA_INL
