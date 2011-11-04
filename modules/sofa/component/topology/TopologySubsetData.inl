/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
void TopologySubsetDataImpl <TopologyElementType, VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology, sofa::core::topology::TopologyHandler *_topologyHandler)
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



/// Funtion used to link Data to point Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySubsetDataImpl <TopologyElementType, VecT>::linkToPointDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToPointDataArray();
}

/// Funtion used to link Data to edge Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySubsetDataImpl <TopologyElementType, VecT>::linkToEdgeDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToEdgeDataArray();
}

/// Funtion used to link Data to triangle Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySubsetDataImpl <TopologyElementType, VecT>::linkToTriangleDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToTriangleDataArray();
}

/// Funtion used to link Data to quad Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySubsetDataImpl <TopologyElementType, VecT>::linkToQuadDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToQuadDataArray();
}

/// Funtion used to link Data to tetrahedron Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySubsetDataImpl <TopologyElementType, VecT>::linkToTetrahedronDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToTetrahedronDataArray();
}

/// Funtion used to link Data to hexahedron Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologySubsetDataImpl <TopologyElementType, VecT>::linkToHexahedronDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToHexahedronDataArray();
}



} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSUBSETDATA_INL

