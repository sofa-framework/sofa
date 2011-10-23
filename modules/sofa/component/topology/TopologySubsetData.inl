/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSUBSETDATA_INL
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSUBSETDATA_INL

#include <sofa/component/topology/TopologySubsetData.h>
#include <sofa/component/topology/TopologyEngine.inl>
#include <sofa/component/topology/TopologySubsetDataHandler.inl>

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
TopologySubsetDataImpl <TopologyElementType, VecT>::~TopologySubsetDataImpl()
{
    if (this->m_topologyHandler)
        delete m_topologyHandler;
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetDataImpl <TopologyElementType, VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology, sofa::core::topology::TopologySubsetDataHandler *_topologyHandler)
{
    if (_topology)
    {
        this->m_topologicalEngine = new TopologyEngineImpl<VecT>((sofa::component::topology::TopologySubsetDataImpl<TopologyElementType, VecT>*)this, _topology, _topologyHandler);
        this->linkToElementDataArray();
    }
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetDataImpl <TopologyElementType, VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
    if (_topology)
    {
        this->m_topologyHandler = new TopologySubsetDataHandler<TopologyElementType, VecT>(this);
        this->m_topologicalEngine = new TopologyEngineImpl<VecT>((sofa::component::topology::TopologySubsetDataImpl<TopologyElementType, VecT>*)this, _topology, m_topologyHandler);
        this->linkToElementDataArray();
    }
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetDataImpl <TopologyElementType, VecT>::registerTopologicalData()
{
    if (this->m_topologicalEngine)
        this->m_topologicalEngine->registerTopology();
#ifndef NDEBUG // too much warnings
    else
        std::cout<<"Error: TopologySubsetDataImpl: " << this->getName() << " has no engine. Use createTopologicalEngine function before." << std::endl;
#endif
}

template <typename TopologyElementType, typename VecT>
void TopologySubsetDataImpl <TopologyElementType, VecT>::addInputData(sofa::core::objectmodel::BaseData *_data)
{
    if (this->m_topologicalEngine)
        this->m_topologicalEngine->addInput(_data);
#ifndef NDEBUG // too much warnings
    else
        std::cout<<"Error: TopologySubsetDataImpl: " << this->getName() << " has no engine. Use createTopologicalEngine function before." << std::endl;
#endif
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSUBSETDATA_INL

