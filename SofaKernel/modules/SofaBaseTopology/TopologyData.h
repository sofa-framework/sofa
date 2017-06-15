/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_H
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_H
#include "config.h"

#include <sofa/helper/vector.h>

#include <sofa/core/topology/BaseTopologyData.h>
#include <SofaBaseTopology/TopologyEngine.h>
#include <SofaBaseTopology/TopologyDataHandler.h>



namespace sofa
{

namespace component
{

namespace topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing Edge related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: Edges added, removed, fused, renumbered).
*/
template< class TopologyElementType, class VecT>
class TopologyDataImpl : public sofa::core::topology::BaseTopologyData<VecT>
{

public:
    typedef VecT container_type;
    typedef typename container_type::value_type value_type;

    /// size_type
    typedef typename container_type::size_type size_type;
    /// reference to a value (read-write)
    typedef typename container_type::reference reference;
    /// const reference to a value (read only)
    typedef typename container_type::const_reference const_reference;
    /// const iterator
    typedef typename container_type::const_iterator const_iterator;


    /// Constructor
    TopologyDataImpl( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : sofa::core::topology::BaseTopologyData< VecT >(data),
          m_topologicalEngine(NULL),
          m_topology(NULL),
          m_topologyHandler(NULL)
    {}

    virtual ~TopologyDataImpl();


    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    virtual void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology, sofa::component::topology::TopologyDataHandler<TopologyElementType,VecT>* _topologyHandler, bool deleteHandler = false);

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    virtual void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

    /// Allow to add additionnal dependencies to others Data.
    void addInputData(sofa::core::objectmodel::BaseData* _data);

    /// Function to link the topological Data with the engine and the current topology. And init everything.
    /// This function should be used at the end of the all declaration link to this Data while using it in a component.
    void registerTopologicalData();


    value_type& operator[](int i)
    {
        container_type& data = *(this->beginEdit());
        value_type& result = data[i];
        this->endEdit();
        return result;
    }


    /// Link Data to topology arrays
    void linkToPointDataArray();
    void linkToEdgeDataArray();
    void linkToTriangleDataArray();
    void linkToQuadDataArray();
    void linkToTetrahedronDataArray();
    void linkToHexahedronDataArray();

    sofa::component::topology::TopologyEngineImpl<VecT>* getTopologicalEngine()
    {
        return m_topologicalEngine.get();
    }

    sofa::core::topology::BaseMeshTopology* getTopology()
    {
        return m_topology;
    }

    sofa::component::topology::TopologyDataHandler<TopologyElementType,VecT>* getTopologyHandler()
    {
        return m_topologyHandler;
    }

protected:
    virtual void linkToElementDataArray() {}

    typename sofa::component::topology::TopologyEngineImpl<VecT>::SPtr m_topologicalEngine;
    sofa::core::topology::BaseMeshTopology* m_topology;
    sofa::component::topology::TopologyDataHandler<TopologyElementType,VecT>* m_topologyHandler;
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////   Point Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class PointData : public TopologyDataImpl<core::topology::BaseMeshTopology::Point, VecT>
{
public:
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Point, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Point, VecT>::value_type value_type;

    PointData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Point, VecT>(data)
    {}

protected:
    void linkToElementDataArray() {this->linkToPointDataArray();}
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Edge Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class EdgeData : public TopologyDataImpl<core::topology::BaseMeshTopology::Edge, VecT>
{
public:
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Edge, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Edge, VecT>::value_type value_type;

    EdgeData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Edge, VecT>(data)
    {}

protected:
    void linkToElementDataArray() {this->linkToEdgeDataArray();}

};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Triangle Topology Data Implementation   ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class TriangleData : public TopologyDataImpl<core::topology::BaseMeshTopology::Triangle, VecT>
{
public:
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Triangle, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Triangle, VecT>::value_type value_type;

    TriangleData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Triangle, VecT>(data)
    {}

protected:
    void linkToElementDataArray() {this->linkToTriangleDataArray();}

};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Quad Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class QuadData : public TopologyDataImpl<core::topology::BaseMeshTopology::Quad, VecT>
{
public:
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Quad, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Quad, VecT>::value_type value_type;

    QuadData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Quad, VecT>(data)
    {}

protected:
    void linkToElementDataArray() {this->linkToQuadDataArray();}

};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Tetrahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class TetrahedronData : public TopologyDataImpl<core::topology::BaseMeshTopology::Tetrahedron, VecT>
{
public:
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Tetrahedron, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Tetrahedron, VecT>::value_type value_type;

    TetrahedronData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Tetrahedron, VecT>(data)
    {}

protected:
    void linkToElementDataArray() {this->linkToTetrahedronDataArray();}

};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Hexahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class HexahedronData : public TopologyDataImpl<core::topology::BaseMeshTopology::Hexahedron, VecT>
{
public:
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Hexahedron, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<core::topology::BaseMeshTopology::Hexahedron, VecT>::value_type value_type;

    HexahedronData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Hexahedron, VecT>(data)
    {}

protected:
    void linkToElementDataArray() {this->linkToHexahedronDataArray();}

};


} // namespace topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_H
