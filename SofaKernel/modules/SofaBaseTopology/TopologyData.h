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

/** \brief A class for storing topology related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: element added, removed, fused, renumbered).
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


    const value_type& operator[](int i) const
    {
        const container_type& data = *(this->getValue());
        const value_type& result = data[i];
        return result;
    }

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

    typename sofa::component::topology::TopologyEngineImpl<VecT>::SPtr m_topologicalEngine;
    sofa::core::topology::BaseMeshTopology* m_topology;
    sofa::component::topology::TopologyDataHandler<TopologyElementType,VecT>* m_topologyHandler;

    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Point*      ) { linkToPointDataArray();       }
    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Edge*       ) { linkToEdgeDataArray();        }
    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Triangle*   ) { linkToTriangleDataArray();    }
    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Quad*       ) { linkToQuadDataArray();        }
    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Tetrahedron*) { linkToTetrahedronDataArray(); }
    void linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Hexahedron* ) { linkToHexahedronDataArray();  }

};

// In c++11, the following classes could be replaced by :
// template< class VecT > using PointData       = TopologyDataImpl<Point      , VecT>;
// template< class VecT > using EdgeData        = TopologyDataImpl<Edge       , VecT>;
// template< class VecT > using TriangleData    = TopologyDataImpl<Triangle   , VecT>;
// template< class VecT > using QuadData        = TopologyDataImpl<Quad       , VecT>;
// template< class VecT > using TetrahedronData = TopologyDataImpl<Tetrahedron, VecT>;
// template< class VecT > using HexahedronData  = TopologyDataImpl<Hexahedron , VecT>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////   Point Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class PointData : public TopologyDataImpl<core::topology::BaseMeshTopology::Point, VecT>
{
public:
    PointData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Point, VecT>(data)
    {}
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Edge Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class EdgeData : public TopologyDataImpl<core::topology::BaseMeshTopology::Edge, VecT>
{
public:
    EdgeData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Edge, VecT>(data)
    {}
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Triangle Topology Data Implementation   ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class TriangleData : public TopologyDataImpl<core::topology::BaseMeshTopology::Triangle, VecT>
{
public:
    TriangleData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Triangle, VecT>(data)
    {}
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Quad Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class QuadData : public TopologyDataImpl<core::topology::BaseMeshTopology::Quad, VecT>
{
public:
    QuadData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Quad, VecT>(data)
    {}
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Tetrahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class TetrahedronData : public TopologyDataImpl<core::topology::BaseMeshTopology::Tetrahedron, VecT>
{
public:
    TetrahedronData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Tetrahedron, VecT>(data)
    {}
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Hexahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class HexahedronData : public TopologyDataImpl<core::topology::BaseMeshTopology::Hexahedron, VecT>
{
public:
    HexahedronData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<core::topology::BaseMeshTopology::Hexahedron, VecT>(data)
    {}
};


} // namespace topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_H
