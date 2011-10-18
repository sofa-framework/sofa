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
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL

#include <sofa/component/topology/TopologyData.h>

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
void TopologyDataImpl <TopologyElementType, VecT>::registerTopologicalData()
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (this->m_topologicalEngine)
        this->m_topologicalEngine->registerTopology();
#ifndef NDEBUG // too much warnings
    else
        std::cout<<"Error: TopologyDataImpl: " << this->getName() << " has no engine. Use createTopologicalEngine function before." << std::endl;
#endif
#endif
}

template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::addInputData(sofa::core::objectmodel::BaseData *_data)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (this->m_topologicalEngine)
        this->m_topologicalEngine->addInput(_data);
#ifndef NDEBUG // too much warnings
    else
        std::cout<<"Error: TopologyDataImpl: " << this->getName() << " has no engine. Use createTopologicalEngine function before." << std::endl;
#endif
#else
    (void)_data;
#endif
}


/// Funtion used to link Data to point Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::linkToPointDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToPointDataArray();
}

/// Funtion used to link Data to edge Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::linkToEdgeDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToEdgeDataArray();
}

/// Funtion used to link Data to triangle Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::linkToTriangleDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToTriangleDataArray();
}

/// Funtion used to link Data to quad Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::linkToQuadDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToQuadDataArray();
}

/// Funtion used to link Data to tetrahedron Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::linkToTetrahedronDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToTetrahedronDataArray();
}

/// Funtion used to link Data to hexahedron Data array, using the engine's method
template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::linkToHexahedronDataArray()
{
    if(m_topologicalEngine)
        m_topologicalEngine->linkToHexahedronDataArray();
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////   Point Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void PointDataImpl<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new PointSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Edge Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void EdgeDataImpl<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new EdgeSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Triangle Topology Data Implementation   ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename VecT >
void TriangleDataImpl<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new TriangleSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Quad Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void QuadDataImpl<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new QuadSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Tetrahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void TetrahedronDataImpl<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new TetrahedronSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}




////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Hexahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void HexahedronDataImpl<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new HexahedronSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}



} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL
