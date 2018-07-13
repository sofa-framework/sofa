#ifndef CMTOPOLOGYDATA_H
#define CMTOPOLOGYDATA_H

#include "config.h"

#include <sofa/helper/vector.h>

#include <sofa/core/topology/CMBaseTopologyData.h>
#include <SofaBaseTopology/CMTopologyEngine.inl>
#include <SofaBaseTopology/CMTopologyDataHandler.h>



namespace sofa
{

namespace component
{

namespace cm_topology
{


////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing Edge related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: Edges added, removed, fused, renumbered).
*/
template< class TopologyElementType, class T>
class TopologyDataImpl : public sofa::core::cm_topology::BaseTopologyData<T>
{

public:
	using Inherit = sofa::core::cm_topology::BaseTopologyData<T>;
	typedef T value_type;
	using Attribute = typename Inherit::Attribute;
	using container_type = Attribute;

	/// Constructor
	TopologyDataImpl( const typename sofa::core::cm_topology::BaseTopologyData< T >::InitData& data)
		: sofa::core::cm_topology::BaseTopologyData< T >(data),
		  m_topologicalEngine(NULL),
		  m_topology(NULL),
		  m_topologyHandler(NULL)
	{}

	virtual ~TopologyDataImpl();


	/** Public functions to handle topological engine creation */
	/// To create topological engine link to this Data. Pointer to current topology is needed.
	virtual void createTopologicalEngine(sofa::core::topology::MapTopology* _topology, sofa::component::cm_topology::TopologyDataHandler<TopologyElementType,T>* _topologyHandler, bool deleteHandler = false);

	/** Public functions to handle topological engine creation */
	/// To create topological engine link to this Data. Pointer to current topology is needed.
	virtual void createTopologicalEngine(sofa::core::topology::MapTopology* _topology);

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

	sofa::component::cm_topology::TopologyEngineImpl<T>* getTopologicalEngine()
	{
		return m_topologicalEngine.get();
	}

	sofa::core::topology::MapTopology* getTopology()
	{
		return m_topology;
	}

	sofa::component::cm_topology::TopologyDataHandler<TopologyElementType,T>* getTopologyHandler()
	{
		return m_topologyHandler;
	}

protected:
	virtual void linkToElementDataArray() {}

	typename sofa::component::cm_topology::TopologyEngineImpl<T>::SPtr m_topologicalEngine;
	sofa::core::topology::MapTopology* m_topology;
	sofa::component::cm_topology::TopologyDataHandler<TopologyElementType,T>* m_topologyHandler;
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////   Point Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T >
class PointData : public TopologyDataImpl<core::topology::MapTopology::Vertex, T>
{
public:
	typedef typename TopologyDataImpl<core::topology::MapTopology::Vertex, T>::container_type container_type;
	typedef typename TopologyDataImpl<core::topology::MapTopology::Vertex, T>::value_type value_type;

	PointData( const typename sofa::core::cm_topology::BaseTopologyData< T >::InitData& data)
		: TopologyDataImpl<core::topology::MapTopology::Vertex, T>(data)
	{}

protected:
	void linkToElementDataArray() {this->linkToPointDataArray();}
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Edge Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T >
class EdgeData : public TopologyDataImpl<core::topology::MapTopology::Edge, T>
{
public:
	typedef typename TopologyDataImpl<core::topology::MapTopology::Edge, T>::container_type container_type;
	typedef typename TopologyDataImpl<core::topology::MapTopology::Edge, T>::value_type value_type;

	EdgeData( const typename sofa::core::cm_topology::BaseTopologyData< T >::InitData& data)
		: TopologyDataImpl<core::topology::MapTopology::Edge, T>(data)
	{}

protected:
	void linkToElementDataArray() {this->linkToEdgeDataArray();}

};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Triangle Topology Data Implementation   ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T >
class TriangleData : public TopologyDataImpl<core::topology::MapTopology::Face, T>
{
public:
	typedef typename TopologyDataImpl<core::topology::MapTopology::Face, T>::container_type container_type;
	typedef typename TopologyDataImpl<core::topology::MapTopology::Face, T>::value_type value_type;

	TriangleData( const typename sofa::core::cm_topology::BaseTopologyData< T >::InitData& data)
		: TopologyDataImpl<core::topology::MapTopology::Face, T>(data)
	{}

protected:
	void linkToElementDataArray() {this->linkToTriangleDataArray();}

};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Tetrahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T >
class TetrahedronData : public TopologyDataImpl<core::topology::MapTopology::Volume, T>
{
public:
	typedef typename TopologyDataImpl<core::topology::MapTopology::Volume, T>::container_type container_type;
	typedef typename TopologyDataImpl<core::topology::MapTopology::Volume, T>::value_type value_type;

	TetrahedronData( const typename sofa::core::cm_topology::BaseTopologyData< T >::InitData& data)
		: TopologyDataImpl<core::topology::MapTopology::Volume, T>(data)
	{}

protected:
	void linkToElementDataArray() {this->linkToTetrahedronDataArray();}

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Hexahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T >
class HexahedronData : public TopologyDataImpl<core::topology::MapTopology::Volume, T>
{
public:
	typedef typename TopologyDataImpl<core::topology::MapTopology::Volume, T>::container_type container_type;
	typedef typename TopologyDataImpl<core::topology::MapTopology::Volume, T>::value_type value_type;

	HexahedronData( const typename sofa::core::cm_topology::BaseTopologyData< T >::InitData& data)
		: TopologyDataImpl<core::topology::MapTopology::Volume, T>(data)
	{}

protected:
	void linkToElementDataArray() {this->linkToHexahedronDataArray();}

};

} // namespace cm_topology

} // namespace component

} // namespace sofa


#endif // CMTOPOLOGYDATA_H
