#ifndef CMTOPOLOGYHANDLER_H
#define CMTOPOLOGYHANDLER_H

#include <sofa/core/topology/CMTopologyChange.h>

namespace sofa
{

namespace core
{

namespace cm_topology
{

typedef topology::MapTopology::Vertex           Vertex;
typedef topology::MapTopology::Edge             Edge;
typedef topology::MapTopology::Face             Face;
typedef topology::MapTopology::Volume           Volume;


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Generic Handling of Topology Event    /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SOFA_CORE_API TopologyHandler
{
public:
	TopologyHandler() : lastElementIndex(0) {}

	virtual ~TopologyHandler() {}

	virtual void ApplyTopologyChanges(const std::list< const core::cm_topology::TopologyChange *>& _topologyChangeEvents, const unsigned int _dataSize);

	virtual void ApplyTopologyChange(const core::cm_topology::EndingEvent* /*event*/) {}

	///////////////////////// Functions on Points //////////////////////////////////////
	/// Apply swap between point indicPointes elements.
	virtual void ApplyTopologyChange(const core::cm_topology::PointsIndicesSwap* /*event*/) {}
	/// Apply adding points elements.
	virtual void ApplyTopologyChange(const core::cm_topology::PointsAdded* /*event*/) {}
	/// Apply removing points elements.
	virtual void ApplyTopologyChange(const core::cm_topology::PointsRemoved* /*event*/) {}
	/// Apply renumbering on points elements.
	virtual void ApplyTopologyChange(const core::cm_topology::PointsRenumbering* /*event*/) {}
	/// Apply moving points elements.
	virtual void ApplyTopologyChange(const core::cm_topology::PointsMoved* /*event*/) {}

	///////////////////////// Functions on Edges //////////////////////////////////////
	/// Apply swap between edges indices elements.
	virtual void ApplyTopologyChange(const core::cm_topology::EdgesIndicesSwap* /*event*/) {}
	/// Apply adding edges elements.
	virtual void ApplyTopologyChange(const core::cm_topology::EdgesAdded* /*event*/) {}
	/// Apply removing edges elements.
	virtual void ApplyTopologyChange(const core::cm_topology::EdgesRemoved* /*event*/) {}
	/// Apply removing function on moved edges elements.
	virtual void ApplyTopologyChange(const core::cm_topology::EdgesMoved_Removing* /*event*/) {}
	/// Apply adding function on moved edges elements.
	virtual void ApplyTopologyChange(const core::cm_topology::EdgesMoved_Adding* /*event*/) {}
	/// Apply renumbering on edges elements.
	virtual void ApplyTopologyChange(const core::cm_topology::EdgesRenumbering* /*event*/) {}

	///////////////////////// Functions on Triangles //////////////////////////////////////
	/// Apply swap between triangles indices elements.
	virtual void ApplyTopologyChange(const core::cm_topology::FacesIndicesSwap* /*event*/) {}
	/// Apply adding triangles elements.
	virtual void ApplyTopologyChange(const core::cm_topology::FacesAdded* /*event*/) {}
	/// Apply removing triangles elements.
	virtual void ApplyTopologyChange(const core::cm_topology::FacesRemoved* /*event*/) {}
	/// Apply removing function on moved triangles elements.
	virtual void ApplyTopologyChange(const core::cm_topology::FacesMoved_Removing* /*event*/) {}
	/// Apply adding function on moved triangles elements.
	virtual void ApplyTopologyChange(const core::cm_topology::FacesMoved_Adding* /*event*/) {}
	/// Apply renumbering on triangles elements.
	virtual void ApplyTopologyChange(const core::cm_topology::FacesRenumbering* /*event*/) {}


	///////////////////////// Functions on Tetrahedron //////////////////////////////////////
	/// Apply swap between tetrahedron indices elements.
	virtual void ApplyTopologyChange(const core::cm_topology::VolumesIndicesSwap* /*event*/) {}
	/// Apply adding tetrahedron elements.
	virtual void ApplyTopologyChange(const core::cm_topology::VolumesAdded* /*event*/) {}
	/// Apply removing tetrahedron elements.
	virtual void ApplyTopologyChange(const core::cm_topology::VolumesRemoved* /*event*/) {}
	/// Apply removing function on moved tetrahedron elements.
	virtual void ApplyTopologyChange(const core::cm_topology::VolumesMoved_Removing* /*event*/) {}
	/// Apply adding function on moved tetrahedron elements.
	virtual void ApplyTopologyChange(const core::cm_topology::VolumesMoved_Adding* /*event*/) {}
	/// Apply renumbering on tetrahedron elements.
	virtual void ApplyTopologyChange(const core::cm_topology::VolumesRenumbering* /*event*/) {}


	virtual bool isTopologyDataRegistered() {return false;}

	/// Swaps values at indices i1 and i2.
	virtual void swap( unsigned int /*i1*/, unsigned int /*i2*/ ) {}

	/// Reorder the values.
	virtual void renumber( const sofa::helper::vector<unsigned int> &/*index*/ ) {}

protected:
	/// to handle PointSubsetData
	void setDataSetArraySize(const unsigned int s) { lastElementIndex = s-1; }

	/// to handle properly the removal of items, the container must know the index of the last element
	unsigned int lastElementIndex;
};


} // namespace cm_topology

} // namespace core

} // namespace sofa

#endif // CMTOPOLOGYHANDLER_H
