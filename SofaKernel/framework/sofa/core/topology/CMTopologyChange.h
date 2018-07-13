#ifndef CMTOPOLOGYCHANGE_H
#define CMTOPOLOGYCHANGE_H

#include <sofa/core/topology/MapTopology.h>
#include <iostream>

namespace sofa
{

namespace core
{

namespace cm_topology
{


/// The enumeration used to give unique identifiers to TopologyChange objects.
enum TopologyChangeType
{
	BASE,                      ///< For TopologyChange class, should never be used.
	ENDING_EVENT,              ///< To notify the end for the current sequence of topological change events

	POINTSINDICESSWAP,         ///< For PointsIndicesSwap class.
	POINTSADDED,               ///< For PointsAdded class.
	POINTSREMOVED,             ///< For PointsRemoved class.
	POINTSMOVED,               ///< For PointsMoved class.
	POINTSRENUMBERING,         ///< For PointsRenumbering class.

	EDGESINDICESSWAP,          ///< For EdgesIndicesSwap class.
	EDGESADDED,                ///< For EdgesAdded class.
	EDGESREMOVED,              ///< For EdgesRemoved class.
	EDGESMOVED_REMOVING,       ///< For EdgesMoved class (event before changing state).
	EDGESMOVED_ADDING,         ///< For EdgesMoved class.
	EDGESRENUMBERING,          ///< For EdgesRenumbering class.

	FACESINDICESSWAP,      ///< For TrianglesIndicesSwap class.
	FACESADDED,            ///< For TrianglesAdded class.
	FACESREMOVED,          ///< For TrianglesRemoved class.
	FACESMOVED_REMOVING,   ///< For TrianglesMoved class (event before changing state).
	FACESMOVED_ADDING,     ///< For TrianglesMoved class.
	FACESRENUMBERING,      ///< For TrianglesRenumbering class.

	VOLUMESINDICESSWAP,     ///< For TetrahedraIndicesSwap class.
	VOLUMESADDED,           ///< For TetrahedraAdded class.
	VOLUMESREMOVED,         ///< For TetrahedraRemoved class.
	VOLUMESMOVED_REMOVING,  ///< For TetrahedraMoved class (event before changing state).
	VOLUMESMOVED_ADDING,    ///< For TetrahedraMoved class.
	VOLUMESRENUMBERING,     ///< For TetrahedraRenumbering class.

	TOPOLOGYCHANGE_LASTID      ///< user defined topology changes can start here
};

//SOFA_CORE_API TopologyChangeType parseTopologyChangeTypeFromString(const std::string& s); // TODO
//SOFA_CORE_API std::string parseTopologyChangeTypeToString(TopologyChangeType t); // TODO

// forward declarations
class SOFA_CORE_API TopologyChange;
class SOFA_CORE_API EndingEvent;
class SOFA_CORE_API PointsIndicesSwap;
class SOFA_CORE_API PointsAdded;
class SOFA_CORE_API PointsRemoved;
class SOFA_CORE_API PointsMoved;
class SOFA_CORE_API PointsRenumbering;
class SOFA_CORE_API EdgesIndicesSwap;
class SOFA_CORE_API EdgesAdded;
class SOFA_CORE_API EdgesRemoved;
class SOFA_CORE_API EdgesMoved_Removing;
class SOFA_CORE_API EdgesMoved_Adding;
class SOFA_CORE_API EdgesRenumbering;
class SOFA_CORE_API FacesIndicesSwap;
class SOFA_CORE_API FacesAdded;
class SOFA_CORE_API FacesRemoved;
class SOFA_CORE_API FacesMoved_Removing;
class SOFA_CORE_API FacesMoved_Adding;
class SOFA_CORE_API FacesRenumbering;
class SOFA_CORE_API VolumesIndicesSwap;
class SOFA_CORE_API VolumesAdded;
class SOFA_CORE_API VolumesRemoved;
class SOFA_CORE_API VolumesMoved_Removing;
class SOFA_CORE_API VolumesMoved_Adding;
class SOFA_CORE_API VolumesRenumbering;


/// Topology identification of a primitive element
struct TopologyElemID
{
	TopologyElemID() : type(VERTEX), index(UINT_MAX) {}

	TopologyElemID(TopologyObjectType _type, unsigned int _index)
		: type(_type)
		, index(_index)
	{}

	TopologyObjectType type;
	unsigned int index;
};

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const TopologyElemID& d);
SOFA_CORE_API std::istream& operator >> (std::istream& in, TopologyElemID& d);

/// Topology change informations related to the ancestor topology element of a point
struct PointAncestorElem
{
	typedef defaulttype::Vec<3, double> LocalCoords;
	using Vertex = topology::MapTopology::Vertex;

	PointAncestorElem() : type(VERTEX), index(Vertex()) {}

	PointAncestorElem(TopologyObjectType _type, Vertex _index, const LocalCoords& _localCoords)
		: type(_type)
		, index(_index)
		, localCoords(_localCoords)
	{}

	TopologyObjectType type;
	topology::MapTopology::Vertex index;
	LocalCoords localCoords;
};

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const PointAncestorElem& d);
SOFA_CORE_API std::istream& operator >> (std::istream& in, PointAncestorElem& d);

/// Topology change informations related to the ancestor topology element of an edge

struct ElemAncestorElem
{

	ElemAncestorElem()
	{}

	ElemAncestorElem(const helper::vector<PointAncestorElem>& _pointSrcElems,
		const helper::vector<TopologyElemID>& _srcElems)
		: pointSrcElems(_pointSrcElems)
		, srcElems(_srcElems)
	{}

	ElemAncestorElem(const helper::vector<PointAncestorElem>& _pointSrcElems,
		const TopologyElemID& _srcElem)
		: pointSrcElems(_pointSrcElems)
		, srcElems()
	{
		srcElems.push_back(_srcElem);
	}

	helper::vector<PointAncestorElem> pointSrcElems;
	helper::vector<TopologyElemID> srcElems;
};


inline SOFA_CORE_API std::ostream& operator << (std::ostream& out, const ElemAncestorElem& d);

inline SOFA_CORE_API std::istream& operator >> (std::istream& in, ElemAncestorElem& d);

template<class TopologyElement>
struct TopologyChangeElementInfo;

template<>
struct TopologyChangeElementInfo<topology::MapTopology::Vertex>
{
	enum { USE_EMOVED          = 1 };
	enum { USE_EMOVED_REMOVING = 0 };
	enum { USE_EMOVED_ADDING   = 0 };

	typedef PointsIndicesSwap    EIndicesSwap;
	typedef PointsRenumbering    ERenumbering;
	typedef PointsAdded          EAdded;
	typedef PointsRemoved        ERemoved;
	typedef PointsMoved          EMoved;
	/// This event is not used for this type of element
	class EMoved_Removing { };
	/// This event is not used for this type of element
	class EMoved_Adding { };

	typedef PointAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<topology::MapTopology::Edge>
{
	enum { USE_EMOVED          = 0 };
	enum { USE_EMOVED_REMOVING = 1 };
	enum { USE_EMOVED_ADDING   = 1 };

	typedef EdgesIndicesSwap    EIndicesSwap;
	typedef EdgesRenumbering    ERenumbering;
	typedef EdgesAdded          EAdded;
	typedef EdgesRemoved        ERemoved;
	typedef EdgesMoved_Removing EMoved_Removing;
	typedef EdgesMoved_Adding   EMoved_Adding;
	/// This event is not used for this type of element
	class EMoved { };

	typedef ElemAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<topology::MapTopology::Face>
{
	enum { USE_EMOVED          = 0 };
	enum { USE_EMOVED_REMOVING = 1 };
	enum { USE_EMOVED_ADDING   = 1 };

	typedef FacesIndicesSwap    EIndicesSwap;
	typedef FacesRenumbering    ERenumbering;
	typedef FacesAdded          EAdded;
	typedef FacesRemoved        ERemoved;
	typedef FacesMoved_Removing EMoved_Removing;
	typedef FacesMoved_Adding   EMoved_Adding;
	/// This event is not used for this type of element
	class EMoved { };

	typedef ElemAncestorElem AncestorElem;
};


template<>
struct TopologyChangeElementInfo<topology::MapTopology::Volume>
{
	enum { USE_EMOVED          = 0 };
	enum { USE_EMOVED_REMOVING = 1 };
	enum { USE_EMOVED_ADDING   = 1 };

	typedef VolumesIndicesSwap    EIndicesSwap;
	typedef VolumesRenumbering    ERenumbering;
	typedef VolumesAdded          EAdded;
	typedef VolumesRemoved        ERemoved;
	typedef VolumesMoved_Removing EMoved_Removing;
	typedef VolumesMoved_Adding   EMoved_Adding;
	/// This event is not used for this type of element
	class EMoved { };

	typedef ElemAncestorElem AncestorElem;
};

/** \brief Base class to indicate a topology change occurred.
*
* All topological changes taking place in a given BaseTopology will issue a TopologyChange in the
* BaseTopology's changeList, so that BasicTopologies mapped to it can know what happened and decide how to
* react.
* Classes inheriting from this one describe a given topolopy change (e.g. RemovedPoint, AddedEdge, etc).
* The exact type of topology change is given by member changeType.
*/
class SOFA_CORE_API TopologyChange
{
public:
	using Dart		= topology::MapTopology::Dart;
	using Vertex	= topology::MapTopology::Vertex;
	using Edge		= topology::MapTopology::Edge;
	using Face		= topology::MapTopology::Face;
	using Volume	= topology::MapTopology::Volume;

	/** \ brief Destructor.
		*
		* Must be virtual for TopologyChange to be a Polymorphic type.
		*/
	virtual ~TopologyChange();

	/** \brief Returns the code of this TopologyChange. */
	TopologyChangeType getChangeType() const { return m_changeType;}

	virtual bool write(std::ostream& out) const;
	virtual bool read(std::istream& in);

	/// Output  stream
	friend std::ostream& operator<< ( std::ostream& out, const TopologyChange* t )
	{
		if (t)
		{
			t->write(out);
		}
		return out;
	}

	/// Input (empty) stream
	friend std::istream& operator>> ( std::istream& in, TopologyChange*& t )
	{
		if (t)
		{
			t->read(in);
		}
		return in;
	}

	/// Input (empty) stream
	friend std::istream& operator>> ( std::istream& in, const TopologyChange*& )
	{
		return in;
	}

protected:
	TopologyChange( TopologyChangeType changeType = BASE )
		: m_changeType(changeType)
	{}

	TopologyChangeType m_changeType; ///< A code that tells the nature of the Topology modification event (could be an enum).
};

/** notifies the end for the current sequence of topological change events */
class SOFA_CORE_API EndingEvent : public core::cm_topology::TopologyChange
{
public:
	EndingEvent()
		: core::cm_topology::TopologyChange(core::cm_topology::ENDING_EVENT)
	{}

	virtual ~EndingEvent();
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   Point Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two points are being swapped */
class SOFA_CORE_API PointsIndicesSwap : public core::cm_topology::TopologyChange
{
public:
	PointsIndicesSwap(Vertex i1,Vertex i2) : core::cm_topology::TopologyChange(core::cm_topology::POINTSINDICESSWAP)
	{
		index[0] = i1;
		index[1] = i2;
	}

	virtual ~PointsIndicesSwap();

public:
	Vertex index[2];
};


/** indicates that some points were added */
class SOFA_CORE_API PointsAdded : public core::cm_topology::TopologyChange
{
public:

	PointsAdded(const unsigned int nV) : core::cm_topology::TopologyChange(core::cm_topology::POINTSADDED)
		, nVertices(nV)
	{ }

	PointsAdded(const unsigned int nV,
			const sofa::helper::vector< Vertex >& indices)
		: core::cm_topology::TopologyChange(core::cm_topology::POINTSADDED)
		, nVertices(nV), pointIndexArray(indices)
	{ }

	PointsAdded(const unsigned int nV,
			const sofa::helper::vector< sofa::helper::vector< Vertex > >& ancestors,
			const sofa::helper::vector< sofa::helper::vector< double       > >& baryCoefs)
		: core::cm_topology::TopologyChange(core::cm_topology::POINTSADDED)
		, nVertices(nV), ancestorsList(ancestors), coefs(baryCoefs)
	{ }

	PointsAdded(const unsigned int nV,
			const sofa::helper::vector< Vertex >& indices,
			const sofa::helper::vector< sofa::helper::vector< Vertex > >& ancestors,
			const sofa::helper::vector< sofa::helper::vector< double       > >& baryCoefs)
		: core::cm_topology::TopologyChange(core::cm_topology::POINTSADDED)
		, nVertices(nV), pointIndexArray(indices), ancestorsList(ancestors), coefs(baryCoefs)
	{ }

	PointsAdded(const unsigned int nV,
			const sofa::helper::vector< Vertex >& indices,
			const sofa::helper::vector< PointAncestorElem >& srcElems,
			const sofa::helper::vector< sofa::helper::vector< Vertex > >& ancestors,
			const sofa::helper::vector< sofa::helper::vector< double       > >& baryCoefs)
		: core::cm_topology::TopologyChange(core::cm_topology::POINTSADDED)
		, nVertices(nV)
		, pointIndexArray(indices)
		, ancestorsList(ancestors)
		, coefs(baryCoefs)
		, ancestorElems(srcElems)
	{
	}

	virtual ~PointsAdded();

	unsigned int getNbAddedVertices() const {return nVertices;}

	unsigned int getNbAddedElements() const { return nVertices; }
	const sofa::helper::vector< Vertex >& getIndexArray() const { return pointIndexArray; }
	const sofa::helper::vector< Vertex >& getElementArray() const { return pointIndexArray; }


public:
	unsigned int nVertices;
	sofa::helper::vector< Vertex > pointIndexArray;
	sofa::helper::vector< sofa::helper::vector< Vertex > > ancestorsList;
	sofa::helper::vector< sofa::helper::vector< double > > coefs;
	sofa::helper::vector< PointAncestorElem > ancestorElems;
};


/** indicates that some points are about to be removed */
class SOFA_CORE_API PointsRemoved : public core::cm_topology::TopologyChange
{
public:
	PointsRemoved(const sofa::helper::vector<Vertex>& _vArray) : core::cm_topology::TopologyChange(core::cm_topology::POINTSREMOVED),
		removedVertexArray(_vArray)
	{ }

	virtual ~PointsRemoved();

	const sofa::helper::vector<Vertex> &getArray() const { return removedVertexArray;	}

public:
	sofa::helper::vector<Vertex> removedVertexArray;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API PointsRenumbering : public core::cm_topology::TopologyChange
{
public:

	PointsRenumbering() : core::cm_topology::TopologyChange(core::cm_topology::POINTSRENUMBERING)
	{ }

	PointsRenumbering(const sofa::helper::vector< Vertex >& indices,
			const sofa::helper::vector< Vertex >& inv_indices)
		: core::cm_topology::TopologyChange(core::cm_topology::POINTSRENUMBERING),
		  indexArray(indices), inv_indexArray(inv_indices)
	{ }

	virtual ~PointsRenumbering();

	const sofa::helper::vector<Vertex> &getIndexArray() const { return indexArray; }

	const sofa::helper::vector<Vertex> &getinv_IndexArray() const { return inv_indexArray; }

public:
	sofa::helper::vector<Vertex> indexArray;
	sofa::helper::vector<Vertex> inv_indexArray;
};


/** indicates that some points were moved */
class SOFA_CORE_API PointsMoved : public core::cm_topology::TopologyChange
{
public:

	PointsMoved(const sofa::helper::vector<Vertex>& indices,
			const sofa::helper::vector< sofa::helper::vector< Vertex > >& ancestors,
			const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
		: core::cm_topology::TopologyChange(core::cm_topology::POINTSMOVED)
		, indicesList(indices), ancestorsList(ancestors), baryCoefsList(baryCoefs)
	{}

	virtual ~PointsMoved();

	const sofa::helper::vector<Vertex> &getIndexArray() const { return indicesList; }

public:
	sofa::helper::vector<Vertex> indicesList;
	sofa::helper::vector< sofa::helper::vector< Vertex > > ancestorsList;
	sofa::helper::vector< sofa::helper::vector< double > > baryCoefsList;
};





////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Edge Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two edges are being swapped */
class SOFA_CORE_API EdgesIndicesSwap : public core::cm_topology::TopologyChange
{
public:
	EdgesIndicesSwap(Edge i1,Edge i2) : core::cm_topology::TopologyChange(core::cm_topology::EDGESINDICESSWAP)
	{
		index[0]=i1;
		index[1]=i2;
	}

	virtual ~EdgesIndicesSwap();

public:
	Edge index[2];
};


/** indicates that some edges were added */
class SOFA_CORE_API EdgesAdded : public core::cm_topology::TopologyChange
{
public:
	EdgesAdded(const unsigned int nE) : core::cm_topology::TopologyChange(core::cm_topology::EDGESADDED),
		nEdges(nE)
	{ }

	EdgesAdded(const unsigned int nE,
			const sofa::helper::vector< Edge >& edgesList)
		: core::cm_topology::TopologyChange(core::cm_topology::EDGESADDED),
		  nEdges(nE),
		  edgeArray(edgesList)
	{ }

	EdgesAdded(const unsigned int nE,
			const sofa::helper::vector< Edge >& edgesList,
			const sofa::helper::vector< sofa::helper::vector< Edge > >& ancestors)
		: core::cm_topology::TopologyChange(core::cm_topology::EDGESADDED),
		  nEdges(nE),
		  edgeArray(edgesList),
		  ancestorsList(ancestors)
	{ }

	EdgesAdded(const unsigned int nE,
			const sofa::helper::vector< Edge >& edgesList,
			const sofa::helper::vector< sofa::helper::vector< Edge > >& ancestors,
			const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
		: core::cm_topology::TopologyChange(core::cm_topology::EDGESADDED),
		  nEdges(nE),
		  edgeArray(edgesList),
		  ancestorsList(ancestors),
		  coefs(baryCoefs)
	{ }

	EdgesAdded(const unsigned int nE,
			const sofa::helper::vector< Edge >& edgesList,
			const sofa::helper::vector< ElemAncestorElem >& srcElems,
			const sofa::helper::vector< sofa::helper::vector< Edge > >& ancestors,
			const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
		: core::cm_topology::TopologyChange(core::cm_topology::EDGESADDED),
		  nEdges(nE),
		  edgeArray(edgesList),
		  ancestorsList(ancestors),
		  coefs(baryCoefs)
		, ancestorElems(srcElems)
	{ }

	virtual ~EdgesAdded();

	unsigned int getNbAddedEdges() const { return nEdges;}
	/*	const sofa::helper::vector<unsigned int> &getArray() const
		{
				return edgeIndexArray;
		}*/
	const sofa::helper::vector< Edge > &getArray() const
	{
		return edgeArray;
	}

	unsigned int getNbAddedElements() const { return nEdges; }
	const sofa::helper::vector< Edge >& getIndexArray() const { return edgeArray; }
	const sofa::helper::vector< Edge >& getElementArray() const { return edgeArray; }

public:
	unsigned int nEdges;
	sofa::helper::vector< Edge > edgeArray;
	sofa::helper::vector< sofa::helper::vector< Edge > > ancestorsList;
	sofa::helper::vector< sofa::helper::vector< double > > coefs;
	sofa::helper::vector< ElemAncestorElem > ancestorElems;
};


/** indicates that some edges are about to be removed */
class SOFA_CORE_API EdgesRemoved : public core::cm_topology::TopologyChange
{
public:
	EdgesRemoved(const sofa::helper::vector<Edge> _eArray) : core::cm_topology::TopologyChange(core::cm_topology::EDGESREMOVED),
		removedEdgesArray(_eArray)
	{}

	virtual ~EdgesRemoved();

	virtual const sofa::helper::vector<Edge> &getArray() const
	{
		return removedEdgesArray;
	}

	virtual std::size_t getNbRemovedEdges() const
	{
		return removedEdgesArray.size();
	}

public:
	sofa::helper::vector<Edge> removedEdgesArray;
};


/** indicates that some edges are about to be moved (i.e one or both of their vertices have just been moved)
 * EdgesMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API EdgesMoved_Removing : public core::cm_topology::TopologyChange
{
public:
	EdgesMoved_Removing (const sofa::helper::vector< Edge >& edgeShell) : core::cm_topology::TopologyChange (core::cm_topology::EDGESMOVED_REMOVING),
		edgesAroundVertexMoved (edgeShell)
	{}

	virtual ~EdgesMoved_Removing();

	const sofa::helper::vector< Edge >& getIndexArray() const { return edgesAroundVertexMoved; }

public:
	sofa::helper::vector< Edge > edgesAroundVertexMoved;
};


/** indicates that some edges are about to be moved (i.e one or both of their vertices have just been moved)
 * EdgesMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API EdgesMoved_Adding : public core::cm_topology::TopologyChange
{
public:
	EdgesMoved_Adding (const sofa::helper::vector< Edge >& edgeShell,
			const sofa::helper::vector< Edge >& edgeArray)
		: core::cm_topology::TopologyChange (core::cm_topology::EDGESMOVED_ADDING),
		  edgesAroundVertexMoved (edgeShell), edgeArray2Moved (edgeArray)
	{}

	virtual ~EdgesMoved_Adding();

	const sofa::helper::vector< Edge >& getIndexArray() const { return edgesAroundVertexMoved; }
	const sofa::helper::vector< Edge >& getElementArray() const { return edgeArray2Moved; }

public:
	sofa::helper::vector< Edge > edgesAroundVertexMoved;
	sofa::helper::vector< topology::MapTopology::Edge > edgeArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API EdgesRenumbering : public core::cm_topology::TopologyChange
{
public:
	EdgesRenumbering() : core::cm_topology::TopologyChange(core::cm_topology::EDGESRENUMBERING)
	{ }

	EdgesRenumbering(const sofa::helper::vector< Edge >& indices,
			const sofa::helper::vector< Edge >& inv_indices)
		: core::cm_topology::TopologyChange(core::cm_topology::EDGESRENUMBERING),
		  indexArray(indices), inv_indexArray(inv_indices)
	{ }

	virtual ~EdgesRenumbering();

	const sofa::helper::vector<Edge> &getIndexArray() const { return indexArray; }

	const sofa::helper::vector<Edge> &getinv_IndexArray() const { return inv_indexArray; }

public:
	sofa::helper::vector<Edge> indexArray;
	sofa::helper::vector<Edge> inv_indexArray;
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Triangle Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two Triangles are being swapped */
class SOFA_CORE_API FacesIndicesSwap : public core::cm_topology::TopologyChange
{
public:
	FacesIndicesSwap(Face i1,Face i2) : core::cm_topology::TopologyChange(core::cm_topology::FACESINDICESSWAP)
	{
		index[0]=i1;
		index[1]=i2;
	}

	virtual ~FacesIndicesSwap();

public:
	Face index[2];
};


/** indicates that some triangles were added */
class SOFA_CORE_API FacesAdded : public core::cm_topology::TopologyChange
{
public:
	FacesAdded(const unsigned int nT) : core::cm_topology::TopologyChange(core::cm_topology::FACESADDED),
		nFaces(nT)
	{ }

	FacesAdded(const unsigned int nT,
			const sofa::helper::vector< Face >& _triangleArray)
		: core::cm_topology::TopologyChange(core::cm_topology::FACESADDED),
		  nFaces(nT),
		  faceArray(_triangleArray)
	{ }

	FacesAdded(const unsigned int nT,
			const sofa::helper::vector< Face >& _triangleArray,
			const sofa::helper::vector< sofa::helper::vector< Face > >& ancestors,
			const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
		: core::cm_topology::TopologyChange(core::cm_topology::FACESADDED),
		  nFaces(nT),
		  faceArray(_triangleArray),
		  ancestorsList(ancestors),
		  coefs(baryCoefs)
	{ }

	FacesAdded(const unsigned int nT,
			const sofa::helper::vector< Face >& _triangleArray,
			const sofa::helper::vector< ElemAncestorElem >& srcElems,
			const sofa::helper::vector< sofa::helper::vector< Face > >& ancestors,
			const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
		: core::cm_topology::TopologyChange(core::cm_topology::FACESADDED),
		  nFaces(nT),
		  faceArray(_triangleArray),
		  ancestorsList(ancestors),
		  coefs(baryCoefs)
		, ancestorElems(srcElems)
	{ }

	virtual ~FacesAdded();

	unsigned int getNbAddedFaces() const
	{
		return nFaces;
	}

	const sofa::helper::vector<Face> &getArray() const
	{
		return faceArray;
	}

	const topology::MapTopology::Face &getFace(const unsigned int i)
	{
		return faceArray[i];
	}

	unsigned int getNbAddedElements() const { return nFaces; }
	const sofa::helper::vector< Face >& getElementArray() const { return faceArray; }

public:
	unsigned int nFaces;
	sofa::helper::vector< Face > faceArray;
	sofa::helper::vector< sofa::helper::vector< Face > > ancestorsList;
	sofa::helper::vector< sofa::helper::vector< double > > coefs;
	sofa::helper::vector< ElemAncestorElem > ancestorElems;
};


/** indicates that some triangles are about to be removed */
class SOFA_CORE_API FacesRemoved : public core::cm_topology::TopologyChange
{
public:
	FacesRemoved(const sofa::helper::vector<Face> _tArray) : core::cm_topology::TopologyChange(core::cm_topology::FACESREMOVED),
		removedFacesArray(_tArray)
	{}

	virtual ~FacesRemoved();

	std::size_t getNbRemovedFaces() const
	{
		return removedFacesArray.size();
	}

	const sofa::helper::vector<Face> &getArray() const
	{
		return removedFacesArray;
	}

protected:
	sofa::helper::vector<Face> removedFacesArray;
};


/** indicates that some triangles are about to be moved (i.e some/all of their vertices have just been moved)
 * TrianglesMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API FacesMoved_Removing : public core::cm_topology::TopologyChange
{
public:
	FacesMoved_Removing (const sofa::helper::vector< Face >& triangleShell)
		: core::cm_topology::TopologyChange (core::cm_topology::FACESMOVED_REMOVING),
		  facesAroundVertexMoved (triangleShell)
	{}

	virtual ~FacesMoved_Removing();

	const sofa::helper::vector< Face >& getIndexArray() const { return facesAroundVertexMoved; }

public:
	sofa::helper::vector< Face > facesAroundVertexMoved;
};


/** indicates that some triangles are about to be moved (i.e some/all of their vertices have just been moved)
 * TrianglesMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API FacesMoved_Adding : public core::cm_topology::TopologyChange
{
public:
	FacesMoved_Adding (const sofa::helper::vector< Face >& triangleShell,
			const sofa::helper::vector< Face >& triangleArray)
		: core::cm_topology::TopologyChange (core::cm_topology::FACESMOVED_ADDING),
		  facesAroundVertexMoved (triangleShell), faceArray2Moved (triangleArray)
	{}

	virtual ~FacesMoved_Adding();

	const sofa::helper::vector< Face >& getElementArray() const { return faceArray2Moved; }

public:
	sofa::helper::vector< Face > facesAroundVertexMoved;
	const sofa::helper::vector< Face > faceArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API FacesRenumbering : public core::cm_topology::TopologyChange
{
public:

	FacesRenumbering() : core::cm_topology::TopologyChange(core::cm_topology::FACESRENUMBERING)
	{ }

	FacesRenumbering(const sofa::helper::vector< Face >& indices,
			const sofa::helper::vector< Face >& inv_indices)
		: core::cm_topology::TopologyChange(core::cm_topology::FACESRENUMBERING),
		  indexArray(indices), inv_indexArray(inv_indices)
	{ }

	virtual ~FacesRenumbering();

	const sofa::helper::vector<Face> &getIndexArray() const { return indexArray; }

	const sofa::helper::vector<Face> &getinv_IndexArray() const { return inv_indexArray; }

public:
	sofa::helper::vector<Face> indexArray;
	sofa::helper::vector<Face> inv_indexArray;
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Tetrahedron Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two Tetrahedra are being swapped */
class SOFA_CORE_API VolumesIndicesSwap : public core::cm_topology::TopologyChange
{
public:
	VolumesIndicesSwap(Volume i1, Volume i2) : core::cm_topology::TopologyChange(core::cm_topology::VOLUMESINDICESSWAP)
	{
		index[0]=i1;
		index[1]=i2;
	}

	virtual ~VolumesIndicesSwap();

public:
	Volume index[2];
};


/** indicates that some tetrahedra were added */
class SOFA_CORE_API VolumesAdded : public core::cm_topology::TopologyChange
{
public:
	VolumesAdded(const sofa::helper::vector< Volume >& tetrahedronArray)
		: core::cm_topology::TopologyChange(core::cm_topology::VOLUMESADDED),
		  nVolumes(tetrahedronArray.size()),
		  volumeArray(tetrahedronArray)
	{ }

	VolumesAdded(
			const sofa::helper::vector< Volume >& _tetrahedronArray,
			const sofa::helper::vector< sofa::helper::vector< Volume > >& ancestors,
			const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
		: core::cm_topology::TopologyChange(core::cm_topology::VOLUMESADDED),
		  nVolumes(_tetrahedronArray.size()),
		  volumeArray(_tetrahedronArray),
		  ancestorsList(ancestors),
		  coefs(baryCoefs)
	{ }

	VolumesAdded(const unsigned int nT,
			const sofa::helper::vector< Volume >& _tetrahedronArray,
			const sofa::helper::vector< ElemAncestorElem >& srcElems,
			const sofa::helper::vector< sofa::helper::vector< Volume > >& ancestors,
			const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
		: core::cm_topology::TopologyChange(core::cm_topology::VOLUMESADDED),
		  nVolumes(nT),
		  volumeArray(_tetrahedronArray),
		  ancestorsList(ancestors),
		  coefs(baryCoefs),
		  ancestorElems(srcElems)
	{ }

	virtual ~VolumesAdded();

	const sofa::helper::vector<Volume> &getArray() const
	{
		return volumeArray;
	}

	unsigned int getNbAddedVolumes() const
	{
		return nVolumes;
	}

	unsigned int getNbAddedElements() const { return nVolumes; }
	const sofa::helper::vector< Volume >& getElementArray() const { return volumeArray; }

public:
	unsigned int nVolumes;
	sofa::helper::vector< Volume > volumeArray;
	sofa::helper::vector< sofa::helper::vector< Volume > > ancestorsList;
	sofa::helper::vector< sofa::helper::vector< double > > coefs;
	sofa::helper::vector< ElemAncestorElem > ancestorElems;
};

/** indicates that some tetrahedra are about to be removed */
class SOFA_CORE_API VolumesRemoved : public core::cm_topology::TopologyChange
{
public:
	VolumesRemoved(const sofa::helper::vector<Volume> _tArray)
		: core::cm_topology::TopologyChange(core::cm_topology::VOLUMESREMOVED),
		  removedVolumesArray(_tArray)
	{ }

	virtual ~VolumesRemoved();

	const sofa::helper::vector<Volume> &getArray() const
	{
		return removedVolumesArray;
	}

	std::size_t getNbRemovedVolumes() const
	{
		return removedVolumesArray.size();
	}

public:
	sofa::helper::vector<Volume> removedVolumesArray;
};


/** indicates that some tetrahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * TetrahedraMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API VolumesMoved_Removing : public core::cm_topology::TopologyChange
{
public:
	VolumesMoved_Removing (const sofa::helper::vector< Volume >& tetrahedronShell)
		: core::cm_topology::TopologyChange (core::cm_topology::VOLUMESMOVED_REMOVING),
		  volumesAroundVertexMoved (tetrahedronShell)
	{}

	virtual ~VolumesMoved_Removing();

	const sofa::helper::vector< Volume >& getIndexArray() const { return volumesAroundVertexMoved; }

public:
	sofa::helper::vector< Volume > volumesAroundVertexMoved;
};


/** indicates that some tetrahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * TetrahedraMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API VolumesMoved_Adding : public core::cm_topology::TopologyChange
{
public:
	VolumesMoved_Adding (const sofa::helper::vector< Volume >& tetrahedronShell,
			const sofa::helper::vector< Volume >& tetrahedronArray)
		: core::cm_topology::TopologyChange (core::cm_topology::VOLUMESMOVED_ADDING),
		  volumesAroundVertexMoved (tetrahedronShell), volumeArray2Moved (tetrahedronArray)
	{}

	virtual ~VolumesMoved_Adding();

	const sofa::helper::vector< Volume >& getIndexArray() const { return volumesAroundVertexMoved; }
	const sofa::helper::vector< Volume >& getElementArray() const { return volumeArray2Moved; }

public:
	sofa::helper::vector< Volume > volumesAroundVertexMoved;
	const sofa::helper::vector< Volume > volumeArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API VolumesRenumbering : public core::cm_topology::TopologyChange
{
public:

	VolumesRenumbering()
		: core::cm_topology::TopologyChange(core::cm_topology::VOLUMESRENUMBERING)
	{ }

	VolumesRenumbering(const sofa::helper::vector< Volume >& indices,
			const sofa::helper::vector< Volume >& inv_indices)
		: core::cm_topology::TopologyChange(core::cm_topology::VOLUMESRENUMBERING),
		  indexArray(indices), inv_indexArray(inv_indices)
	{ }

	virtual ~VolumesRenumbering();

	const sofa::helper::vector<Volume> &getIndexArray() const { return indexArray; }

	const sofa::helper::vector<Volume> &getinv_IndexArray() const { return inv_indexArray; }

public:
	sofa::helper::vector<Volume> indexArray;
	sofa::helper::vector<Volume> inv_indexArray;
};

} // namespace cm_topology

} // namespace core

} // namespace sofa
#endif // CMTOPOLOGYCHANGE_H
