#ifndef CMBASETOPOLOGYDATA_H
#define CMBASETOPOLOGYDATA_H

#include <sofa/core/topology/MapTopology.h>

namespace sofa
{

namespace core
{

namespace cm_topology
{

template < class T>
class BaseTopologyData : public sofa::core::objectmodel::Data < topology::MapTopology::Attribute_T<T> >
{
public:
	using Attribute = topology::MapTopology::Attribute_T<T>;
	using DataAttribute = sofa::core::objectmodel::Data<Attribute>;
	using index_type = topology::MapTopology::index_type;
	using Vertex = topology::MapTopology::Vertex;
	using Edge = topology::MapTopology::Edge;
	using Face = topology::MapTopology::Face;
	using Volume = topology::MapTopology::Volume;

	//SOFA_CLASS(SOFA_TEMPLATE2(BaseTopologyData,T,VecT), SOFA_TEMPLATE(sofa::core::objectmodel::Data, T));

	class InitData : public sofa::core::objectmodel::BaseData::BaseInitData
	{
	public:
		InitData() : value(Attribute()) {}
		InitData(const Attribute& v) : value(v) {}
		InitData(const sofa::core::objectmodel::BaseData::BaseInitData& i) : sofa::core::objectmodel::BaseData::BaseInitData(i), value(Attribute()) {}

		Attribute value;
	};

	/** \copydoc Data(const BaseData::BaseInitData&) */
	explicit BaseTopologyData(const sofa::core::objectmodel::BaseData::BaseInitData& init)
		: DataAttribute(init)
	{
	}

	/** \copydoc Data(const InitData&) */
	explicit BaseTopologyData(const InitData& init)
		: DataAttribute(init)
	{
	}


	/** \copydoc Data(const char*, bool, bool) */
	BaseTopologyData( const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false)
		: DataAttribute(helpMsg, isDisplayed, isReadOnly)
	{

	}

	/** \copydoc Data(const T&, const char*, bool, bool) */
	BaseTopologyData( const Attribute& /*value*/, const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false)
		: DataAttribute(helpMsg, isDisplayed, isReadOnly)
	{
	}


	// Generic methods to apply changes on the Data
	//{
	/// Apply adding points elements.
	virtual void applyCreatePointFunction(const sofa::helper::vector<Vertex>& ) {}
	/// Apply removing points elements.
	virtual void applyDestroyPointFunction(const sofa::helper::vector<Vertex>& ) {}

	/// Apply adding edges elements.
	virtual void applyCreateEdgeFunction(const sofa::helper::vector<Edge>& ) {}
	/// Apply removing edges elements.
	virtual void applyDestroyEdgeFunction(const sofa::helper::vector<Edge>& ) {}

	/// Apply adding triangles elements.
	virtual void applyCreateTriangleFunction(const sofa::helper::vector<Face>& ) {}
	/// Apply removing triangles elements.
	virtual void applyDestroyTriangleFunction(const sofa::helper::vector<Face>& ) {}

	/// Apply adding quads elements.
	virtual void applyCreateQuadFunction(const sofa::helper::vector<Face>& ) {}
	/// Apply removing quads elements.
	virtual void applyDestroyQuadFunction(const sofa::helper::vector<Face>& ) {}

	/// Apply adding tetrahedra elements.
	virtual void applyCreateTetrahedronFunction(const sofa::helper::vector<Volume>& ) {}
	/// Apply removing tetrahedra elements.
	virtual void applyDestroyTetrahedronFunction(const sofa::helper::vector<Volume>& ) {}

	/// Apply adding hexahedra elements.
	virtual void applyCreateHexahedronFunction(const sofa::helper::vector<Volume>& ) {}
	/// Apply removing hexahedra elements.
	virtual void applyDestroyHexahedronFunction(const sofa::helper::vector<Volume>& ) {}
	//}


	virtual void add(
			const sofa::helper::vector< sofa::helper::vector< Vertex > >& ,
			const sofa::helper::vector< sofa::helper::vector< SReal > >& )
	{}

	virtual void add(
			const sofa::helper::vector< sofa::helper::vector< Edge > >& ,
			const sofa::helper::vector< sofa::helper::vector< SReal > >& )
	{}

	virtual void add(
			const sofa::helper::vector< sofa::helper::vector< Face > >& ,
			const sofa::helper::vector< sofa::helper::vector< SReal > >& )
	{}

	virtual void add(
			const sofa::helper::vector< sofa::helper::vector< Volume > >& ,
			const sofa::helper::vector< sofa::helper::vector< SReal > >& )
	{}

	/// Remove the values corresponding to the points removed.
	virtual void remove( const sofa::helper::vector<Vertex>& ) {}

	/// Swaps values at indices i1 and i2.
	virtual void swap( unsigned int , unsigned int ) {}

	/// Reorder the values.
	virtual void renumber( const sofa::helper::vector<index_type>& ) {}

	/// Move a list of points
	virtual void move( const sofa::helper::vector<Vertex>& ,
			const sofa::helper::vector< sofa::helper::vector< Vertex> >& ,
			const sofa::helper::vector< sofa::helper::vector< SReal > >& ) {}

};

} // namespace cm_topology

} // namespace component

} // namespace sofa

#endif // CMBASETOPOLOGYDATA_H
