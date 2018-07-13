#ifndef CMTOPOLOGYELEMENTHANDLER_H
#define CMTOPOLOGYELEMENTHANDLER_H

#include <sofa/core/topology/CMTopologyHandler.h>

namespace sofa
{

namespace core
{

namespace cm_topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Generic Handler Implementation   ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing Edge related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: Edges added, removed, fused, renumbered).
*/
template<class TopologyElementType>
class SOFA_CORE_API TopologyElementHandler : public sofa::core::cm_topology::TopologyHandler
{
public:

	typedef core::cm_topology::TopologyElementInfo<TopologyElementType> ElementInfo;
	typedef core::cm_topology::TopologyChangeElementInfo<TopologyElementType> ChangeElementInfo;

	// Event types (EMoved* are not used for all element types, i.e. Point vs others)
	typedef typename ChangeElementInfo::EIndicesSwap    EIndicesSwap;
	typedef typename ChangeElementInfo::ERenumbering    ERenumbering;
	typedef typename ChangeElementInfo::EAdded          EAdded;
	typedef typename ChangeElementInfo::ERemoved        ERemoved;
	typedef typename ChangeElementInfo::EMoved          EMoved;
	typedef typename ChangeElementInfo::EMoved_Removing EMoved_Removing;
	typedef typename ChangeElementInfo::EMoved_Adding   EMoved_Adding;
	typedef typename ChangeElementInfo::AncestorElem    AncestorElem;

	TopologyElementHandler() : TopologyHandler() {}

	virtual ~TopologyElementHandler() {}


	using TopologyHandler::ApplyTopologyChange;

	/// Apply swap between indices elements.
	virtual void ApplyTopologyChange(const EIndicesSwap* /* event*/)
	{
//		 this->swap(event->index[0], event->index[1]);
		// TODO
	}

	/// Apply adding elements.
	virtual void ApplyTopologyChange(const EAdded* event)
	{
		this->add(/*event->getArray(),*/ event->getElementArray(),
			event->ancestorsList, event->coefs, event->ancestorElems);
	}

	/// Apply removing elements.
	virtual void ApplyTopologyChange(const ERemoved* event)
	{
		this->remove(event->getArray());
	}

	/// Apply renumbering on elements.
	virtual void ApplyTopologyChange(const ERenumbering* /* event*/)
	{
		// TODO
		//this->renumber(event->getIndexArray());
	}

	/// Apply moving elements.
	virtual void ApplyTopologyChange(const EMoved* event);
	/// Apply adding function on moved elements.
	virtual void ApplyTopologyChange(const EMoved_Adding* /* event*/)
	{
		// TODO
//		this->addOnMovedPosition(event->getIndexArray(), event->getElementArray());
	}

	/// Apply removing function on moved elements.
	virtual void ApplyTopologyChange(const EMoved_Removing* /* event*/)
	{
		// TODO
//		this->removeOnMovedPosition(event->getIndexArray());
	}

protected:
	/// Swaps values at indices i1 and i2.
	virtual void swap( TopologyElementType /*i1*/, TopologyElementType /*i2*/ ) {}

	/// Reorder the values.
	virtual void renumber( const sofa::helper::vector<TopologyElementType> &/*index*/ ) {}

	/// Add some values. Values are added at the end of the vector.
	/// This is the (old) version, to be deprecated in favor of the next method
	virtual void add( unsigned int /*nbElements*/,
			const sofa::helper::vector< TopologyElementType >& /*elems*/,
			const sofa::helper::vector< sofa::helper::vector< TopologyElementType > > &/*ancestors*/,
			const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}

	/// Add some values. Values are added at the end of the vector.
	/// This (new) version gives more information for element indices and ancestry
	virtual void add( /*const sofa::helper::vector<unsigned int> & index,*/
			const sofa::helper::vector< TopologyElementType >& elems,
			const sofa::helper::vector< sofa::helper::vector< TopologyElementType > > &ancestors,
			const sofa::helper::vector< sofa::helper::vector< double > >& coefs,
			const sofa::helper::vector< AncestorElem >& /*ancestorElems*/
			)
	{
		// call old method by default
		add((unsigned int)elems.size(), elems, ancestors, coefs);
	}

	/// Remove the values corresponding to the ELement removed.
	virtual void remove( const sofa::helper::vector<TopologyElementType> & /*index */) {}

	/// Move a list of points
	virtual void move( const sofa::helper::vector<Vertex> & /*indexList*/,
			const sofa::helper::vector< sofa::helper::vector< Vertex > >& /*ancestors*/,
			const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}

//    /// Add Element after a displacement of vertices, ie. add element based on previous position topology revision.
//    virtual void addOnMovedPosition(const sofa::helper::vector<TopologyElementType> &/*indexList*/,
//            const sofa::helper::vector< TopologyElementType > & /*elems*/) {}

//    /// Remove Element after a displacement of vertices, ie. add element based on previous position topology revision.
//    virtual void removeOnMovedPosition(const sofa::helper::vector<TopologyElementType> &/*indices*/) {}

};


} // namespace cm_topology

} // namespace core

} // namespace sofa

#endif // CMTOPOLOGYELEMENTHANDLER_H
