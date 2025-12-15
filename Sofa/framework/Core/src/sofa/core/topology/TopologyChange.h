/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/core/topology/Topology.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/list.h>

namespace sofa::core::topology
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

    TRIANGLESINDICESSWAP,      ///< For TrianglesIndicesSwap class.
    TRIANGLESADDED,            ///< For TrianglesAdded class.
    TRIANGLESREMOVED,          ///< For TrianglesRemoved class.
    TRIANGLESMOVED_REMOVING,   ///< For TrianglesMoved class (event before changing state).
    TRIANGLESMOVED_ADDING,     ///< For TrianglesMoved class.
    TRIANGLESRENUMBERING,      ///< For TrianglesRenumbering class.

    TETRAHEDRAINDICESSWAP,     ///< For TetrahedraIndicesSwap class.
    TETRAHEDRAADDED,           ///< For TetrahedraAdded class.
    TETRAHEDRAREMOVED,         ///< For TetrahedraRemoved class.
    TETRAHEDRAMOVED_REMOVING,  ///< For TetrahedraMoved class (event before changing state).
    TETRAHEDRAMOVED_ADDING,    ///< For TetrahedraMoved class.
    TETRAHEDRARENUMBERING,     ///< For TetrahedraRenumbering class.

    QUADSINDICESSWAP,          ///< For QuadsIndicesSwap class.
    QUADSADDED,                ///< For QuadsAdded class.
    QUADSREMOVED,              ///< For QuadsRemoved class.
    QUADSMOVED_REMOVING,       ///< For QuadsMoved class (event before changing state).
    QUADSMOVED_ADDING,         ///< For QuadsMoved class.
    QUADSRENUMBERING,          ///< For QuadsRenumbering class.

    HEXAHEDRAINDICESSWAP,      ///< For HexahedraIndicesSwap class.
    HEXAHEDRAADDED,            ///< For HexahedraAdded class.
    HEXAHEDRAREMOVED,          ///< For HexahedraRemoved class.
    HEXAHEDRAMOVED_REMOVING,   ///< For HexahedraMoved class (event before changing state).
    HEXAHEDRAMOVED_ADDING,     ///< For HexahedraMoved class.
    HEXAHEDRARENUMBERING,      ///< For HexahedraRenumbering class.

    TOPOLOGYCHANGE_LASTID      ///< user defined topology changes can start here
};

SOFA_CORE_API TopologyChangeType parseTopologyChangeTypeFromString(const std::string& s);
SOFA_CORE_API std::string parseTopologyChangeTypeToString(TopologyChangeType t);

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
class SOFA_CORE_API TrianglesIndicesSwap;
class SOFA_CORE_API TrianglesAdded;
class SOFA_CORE_API TrianglesRemoved;
class SOFA_CORE_API TrianglesMoved_Removing;
class SOFA_CORE_API TrianglesMoved_Adding;
class SOFA_CORE_API TrianglesRenumbering;
class SOFA_CORE_API TetrahedraIndicesSwap;
class SOFA_CORE_API TetrahedraAdded;
class SOFA_CORE_API TetrahedraRemoved;
class SOFA_CORE_API TetrahedraMoved_Removing;
class SOFA_CORE_API TetrahedraMoved_Adding;
class SOFA_CORE_API TetrahedraRenumbering;
class SOFA_CORE_API QuadsIndicesSwap;
class SOFA_CORE_API QuadsAdded;
class SOFA_CORE_API QuadsRemoved;
class SOFA_CORE_API QuadsMoved_Removing;
class SOFA_CORE_API QuadsMoved_Adding;
class SOFA_CORE_API QuadsRenumbering;
class SOFA_CORE_API HexahedraIndicesSwap;
class SOFA_CORE_API HexahedraAdded;
class SOFA_CORE_API HexahedraRemoved;
class SOFA_CORE_API HexahedraMoved_Removing;
class SOFA_CORE_API HexahedraMoved_Adding;
class SOFA_CORE_API HexahedraRenumbering;



/// Topology identification of a primitive element
struct TopologyElemID
{
    TopologyElemID() : type(geometry::ElementType::POINT), index((Topology::ElemID)-1) {}

    TopologyElemID(geometry::ElementType _type, Topology::ElemID _index)
        : type(_type)
        , index(_index)
    {}

    geometry::ElementType type;
    Topology::ElemID index;
};

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const TopologyElemID& d);
SOFA_CORE_API std::istream& operator >> (std::istream& in, TopologyElemID& d);

/// Topology change information related to the ancestor topology element of a point
struct PointAncestorElem
{
    typedef type::Vec3 LocalCoords;

    PointAncestorElem() : type(geometry::ElementType::POINT), index(sofa::InvalidID) {}

    PointAncestorElem(geometry::ElementType _type, Topology::ElemID _index, const LocalCoords& _localCoords)
        : type(_type)
        , index(_index)
        , localCoords(_localCoords)
    {}
    
    geometry::ElementType type;
    Topology::ElemID index;
    LocalCoords localCoords;
};

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const PointAncestorElem& d);
SOFA_CORE_API std::istream& operator >> (std::istream& in, PointAncestorElem& d);

/// Topology change information related to the ancestor topology element of an edge
template<int NV>
struct ElemAncestorElem
{

    ElemAncestorElem()
    {}

    ElemAncestorElem(const type::fixed_array<PointAncestorElem,NV>& _pointSrcElems,
        const type::vector<TopologyElemID>& _srcElems)
        : pointSrcElems(_pointSrcElems)
        , srcElems(_srcElems)
    {}
    
    ElemAncestorElem(const type::fixed_array<PointAncestorElem,NV>& _pointSrcElems,
        const TopologyElemID& _srcElem)
        : pointSrcElems(_pointSrcElems)
        , srcElems()
    {
        srcElems.push_back(_srcElem);
    }
    
    type::fixed_array<PointAncestorElem,NV> pointSrcElems;
    type::vector<TopologyElemID> srcElems;
};

template<int NV>
std::ostream& operator << (std::ostream& out, const ElemAncestorElem<NV>& d)
{
    out << d.pointSrcElems << " " << d.srcElems.size() << " " << d.srcElems << "\n";
    return out;
}

template<int NV>
std::istream& operator >> (std::istream& in, ElemAncestorElem<NV>& d)
{
    SOFA_UNUSED(d);
    
    return in;
}

typedef ElemAncestorElem<2> EdgeAncestorElem;
typedef ElemAncestorElem<3> TriangleAncestorElem;
typedef ElemAncestorElem<4> QuadAncestorElem;
typedef ElemAncestorElem<4> TetrahedronAncestorElem;
typedef ElemAncestorElem<8> HexahedronAncestorElem;

template<class TopologyElement>
struct TopologyChangeElementInfo;

template<>
struct TopologyChangeElementInfo<Topology::Point>
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
struct TopologyChangeElementInfo<Topology::Edge>
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
    class EMoved ;

    typedef EdgeAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<Topology::Triangle>
{
    enum { USE_EMOVED          = 0 };
    enum { USE_EMOVED_REMOVING = 1 };
    enum { USE_EMOVED_ADDING   = 1 };

    typedef TrianglesIndicesSwap    EIndicesSwap;
    typedef TrianglesRenumbering    ERenumbering;
    typedef TrianglesAdded          EAdded;
    typedef TrianglesRemoved        ERemoved;
    typedef TrianglesMoved_Removing EMoved_Removing;
    typedef TrianglesMoved_Adding   EMoved_Adding;
    /// This event is not used for this type of element
    class EMoved { };

    typedef TriangleAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<Topology::Quad>
{
    enum { USE_EMOVED          = 0 };
    enum { USE_EMOVED_REMOVING = 1 };
    enum { USE_EMOVED_ADDING   = 1 };

    typedef QuadsIndicesSwap    EIndicesSwap;
    typedef QuadsRenumbering    ERenumbering;
    typedef QuadsAdded          EAdded;
    typedef QuadsRemoved        ERemoved;
    typedef QuadsMoved_Removing EMoved_Removing;
    typedef QuadsMoved_Adding   EMoved_Adding;
    /// This event is not used for this type of element
    class EMoved { };

    typedef QuadAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<Topology::Tetrahedron>
{
    enum { USE_EMOVED          = 0 };
    enum { USE_EMOVED_REMOVING = 1 };
    enum { USE_EMOVED_ADDING   = 1 };

    typedef TetrahedraIndicesSwap    EIndicesSwap;
    typedef TetrahedraRenumbering    ERenumbering;
    typedef TetrahedraAdded          EAdded;
    typedef TetrahedraRemoved        ERemoved;
    typedef TetrahedraMoved_Removing EMoved_Removing;
    typedef TetrahedraMoved_Adding   EMoved_Adding;
    /// This event is not used for this type of element
    class EMoved { };

    typedef TetrahedronAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<Topology::Hexahedron>
{
    enum { USE_EMOVED          = 0 };
    enum { USE_EMOVED_REMOVING = 1 };
    enum { USE_EMOVED_ADDING   = 1 };

    typedef HexahedraIndicesSwap    EIndicesSwap;
    typedef HexahedraRenumbering    ERenumbering;
    typedef HexahedraAdded          EAdded;
    typedef HexahedraRemoved        ERemoved;
    typedef HexahedraMoved_Removing EMoved_Removing;
    typedef HexahedraMoved_Adding   EMoved_Adding;
    /// This event is not used for this type of element
    class EMoved { };

    typedef HexahedronAncestorElem AncestorElem;
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
    SOFA_CORE_API friend std::ostream& operator<< ( std::ostream& out, const TopologyChange* t );

    /// Input (empty) stream
    SOFA_CORE_API friend std::istream& operator>> ( std::istream& in, TopologyChange*& t );

    /// Input (empty) stream
    SOFA_CORE_API friend std::istream& operator>> ( std::istream& in, const TopologyChange*& );

protected:
    TopologyChange( TopologyChangeType changeType = BASE )
        : m_changeType(changeType)
    {}

    TopologyChangeType m_changeType; ///< A code that tells the nature of the Topology modification event (could be an enum).
};

/** notifies the end for the current sequence of topological change events */
class SOFA_CORE_API EndingEvent : public core::topology::TopologyChange
{
public:
    EndingEvent()
        : core::topology::TopologyChange(core::topology::ENDING_EVENT)
    {}

    ~EndingEvent() override;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   Point Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two points are being swapped */
class SOFA_CORE_API PointsIndicesSwap : public core::topology::TopologyChange
{
public:
    PointsIndicesSwap(const Topology::PointID i1,const Topology::PointID i2) : core::topology::TopologyChange(core::topology::POINTSINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    ~PointsIndicesSwap() override;

    Topology::PointID index[2];
};


/** indicates that some points were added */
class SOFA_CORE_API PointsAdded : public core::topology::TopologyChange
{
public:

    PointsAdded(const size_t nV) : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV)
    { }

    PointsAdded(const size_t nV,
            const sofa::type::vector< Topology::PointID >& indices)
        : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV), pointIndexArray(indices)
    { }

    PointsAdded(const size_t nV,
            const sofa::type::vector< sofa::type::vector< Topology::PointID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal       > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV), ancestorsList(ancestors), coefs(baryCoefs)
    { }

    PointsAdded(const size_t nV,
            const sofa::type::vector< Topology::PointID >& indices,
            const sofa::type::vector< sofa::type::vector< Topology::PointID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal       > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV), pointIndexArray(indices), ancestorsList(ancestors), coefs(baryCoefs)
    { }

    PointsAdded(const size_t nV,
            const sofa::type::vector< Topology::PointID >& indices,
            const sofa::type::vector< PointAncestorElem >& srcElems,
            const sofa::type::vector< sofa::type::vector< Topology::PointID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal       > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV)
        , pointIndexArray(indices)
        , ancestorsList(ancestors)
        , coefs(baryCoefs)
        , ancestorElems(srcElems)
    {
    }

    ~PointsAdded() override;

    size_t getNbAddedVertices() const {return nVertices;}

    size_t getNbAddedElements() const { return nVertices; }
    const sofa::type::vector< Topology::PointID >& getIndexArray() const { return pointIndexArray; }
    const sofa::type::vector< Topology::Point >& getElementArray() const { return pointIndexArray; }

    size_t nVertices;
    sofa::type::vector< Topology::PointID > pointIndexArray;
    sofa::type::vector< sofa::type::vector< Topology::PointID > > ancestorsList;
    sofa::type::vector< sofa::type::vector< SReal > > coefs;
    sofa::type::vector< PointAncestorElem > ancestorElems;
};


/** indicates that some points are about to be removed */
class SOFA_CORE_API PointsRemoved : public core::topology::TopologyChange
{
public:
    PointsRemoved(const sofa::type::vector<Topology::PointID>& _vArray) : core::topology::TopologyChange(core::topology::POINTSREMOVED),
        removedVertexArray(_vArray)
    { }

    ~PointsRemoved() override;

    const sofa::type::vector<Topology::PointID> &getArray() const { return removedVertexArray;	}

    sofa::type::vector<Topology::PointID> removedVertexArray;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API PointsRenumbering : public core::topology::TopologyChange
{
public:

    PointsRenumbering() : core::topology::TopologyChange(core::topology::POINTSRENUMBERING)
    { }

    PointsRenumbering(const sofa::type::vector< Topology::PointID >& indices,
            const sofa::type::vector< Topology::PointID >& inv_indices)
        : core::topology::TopologyChange(core::topology::POINTSRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    ~PointsRenumbering() override;

    const sofa::type::vector<Topology::PointID> &getIndexArray() const { return indexArray; }

    const sofa::type::vector<Topology::PointID> &getinv_IndexArray() const { return inv_indexArray; }

    sofa::type::vector<Topology::PointID> indexArray;
    sofa::type::vector<Topology::PointID> inv_indexArray;
};


/** indicates that some points were moved */
class SOFA_CORE_API PointsMoved : public core::topology::TopologyChange
{
public:

    PointsMoved(const sofa::type::vector<Topology::PointID>& indices,
            const sofa::type::vector< sofa::type::vector< Topology::PointID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::POINTSMOVED)
        , indicesList(indices), ancestorsList(ancestors), baryCoefsList(baryCoefs)
    {}

    ~PointsMoved() override;
    
    const sofa::type::vector<Topology::PointID> &getIndexArray() const { return indicesList; }

    sofa::type::vector<Topology::PointID> indicesList;
    sofa::type::vector< sofa::type::vector< Topology::PointID > > ancestorsList;
    sofa::type::vector< sofa::type::vector< SReal > > baryCoefsList;
};





////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Edge Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two edges are being swapped */
class SOFA_CORE_API EdgesIndicesSwap : public core::topology::TopologyChange
{
public:
    EdgesIndicesSwap(const Topology::EdgeID i1,const Topology::EdgeID i2) : core::topology::TopologyChange(core::topology::EDGESINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    ~EdgesIndicesSwap() override;

    Topology::EdgeID index[2];
};


/** indicates that some edges were added */
class SOFA_CORE_API EdgesAdded : public core::topology::TopologyChange
{
public:
    EdgesAdded(const size_t nE) : core::topology::TopologyChange(core::topology::EDGESADDED),
        nEdges(nE)
    { }

    EdgesAdded(const size_t nE,
            const sofa::type::vector< Topology::Edge >& edgesList,
            const sofa::type::vector< Topology::EdgeID >& edgesIndex)
        : core::topology::TopologyChange(core::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex)
    { }

    EdgesAdded(const size_t nE,
            const sofa::type::vector< Topology::Edge >& edgesList,
            const sofa::type::vector< Topology::EdgeID >& edgesIndex,
            const sofa::type::vector< sofa::type::vector< Topology::EdgeID > >& ancestors)
        : core::topology::TopologyChange(core::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex),
          ancestorsList(ancestors)
    { }

    EdgesAdded(const size_t nE,
            const sofa::type::vector< Topology::Edge >& edgesList,
            const sofa::type::vector< Topology::EdgeID >& edgesIndex,
            const sofa::type::vector< sofa::type::vector< Topology::EdgeID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }
    
    EdgesAdded(const size_t nE,
            const sofa::type::vector< Topology::Edge >& edgesList,
            const sofa::type::vector< Topology::EdgeID >& edgesIndex,
            const sofa::type::vector< EdgeAncestorElem >& srcElems,
            const sofa::type::vector< sofa::type::vector< Topology::EdgeID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
        , ancestorElems(srcElems)
    { }

    ~EdgesAdded() override;

    size_t getNbAddedEdges() const { return nEdges;}
    /*	const sofa::type::vector<Topology::EdgeID> &getArray() const
        {
                return edgeIndexArray;
        }*/
    const sofa::type::vector< Topology::Edge > &getArray() const
    {
        return edgeArray;
    }

    size_t getNbAddedElements() const { return nEdges; }
    const sofa::type::vector< Topology::EdgeID >& getIndexArray() const { return edgeIndexArray; }
    const sofa::type::vector< Topology::Edge >& getElementArray() const { return edgeArray; }

    size_t nEdges;
    sofa::type::vector< Topology::Edge > edgeArray;
    sofa::type::vector< Topology::EdgeID > edgeIndexArray;
    sofa::type::vector< sofa::type::vector< Topology::EdgeID > > ancestorsList;
    sofa::type::vector< sofa::type::vector< SReal > > coefs;
    sofa::type::vector< EdgeAncestorElem > ancestorElems;
};


/** indicates that some edges are about to be removed */
class SOFA_CORE_API EdgesRemoved : public core::topology::TopologyChange
{
public:
    EdgesRemoved(const sofa::type::vector<Topology::EdgeID> _eArray) : core::topology::TopologyChange(core::topology::EDGESREMOVED),
        removedEdgesArray(_eArray)
    {}

    ~EdgesRemoved() override;

    virtual const sofa::type::vector<Topology::EdgeID> &getArray() const
    {
        return removedEdgesArray;
    }

    virtual std::size_t getNbRemovedEdges() const
    {
        return removedEdgesArray.size();
    }

    sofa::type::vector<Topology::EdgeID> removedEdgesArray;
};


/** indicates that some edges are about to be moved (i.e one or both of their vertices have just been moved)
 * EdgesMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API EdgesMoved_Removing : public core::topology::TopologyChange
{
public:
    EdgesMoved_Removing (const sofa::type::vector< Topology::EdgeID >& edgeShell) : core::topology::TopologyChange (core::topology::EDGESMOVED_REMOVING),
        edgesAroundVertexMoved (edgeShell)
    {}

    ~EdgesMoved_Removing() override;
    
    const sofa::type::vector< Topology::EdgeID >& getIndexArray() const { return edgesAroundVertexMoved; }

    sofa::type::vector< Topology::EdgeID > edgesAroundVertexMoved;
};


/** indicates that some edges are about to be moved (i.e one or both of their vertices have just been moved)
 * EdgesMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API EdgesMoved_Adding : public core::topology::TopologyChange
{
public:
    EdgesMoved_Adding (const sofa::type::vector< Topology::EdgeID >& edgeShell,
            const sofa::type::vector< Topology::Edge >& edgeArray)
        : core::topology::TopologyChange (core::topology::EDGESMOVED_ADDING),
          edgesAroundVertexMoved (edgeShell), edgeArray2Moved (edgeArray)
    {}

    ~EdgesMoved_Adding() override;
    
    const sofa::type::vector< Topology::EdgeID >& getIndexArray() const { return edgesAroundVertexMoved; }
    const sofa::type::vector< Topology::Edge >& getElementArray() const { return edgeArray2Moved; }

    sofa::type::vector< Topology::EdgeID > edgesAroundVertexMoved;
    sofa::type::vector< Topology::Edge > edgeArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API EdgesRenumbering : public core::topology::TopologyChange
{
public:
    EdgesRenumbering() : core::topology::TopologyChange(core::topology::EDGESRENUMBERING)
    { }

    EdgesRenumbering(const sofa::type::vector< Topology::EdgeID >& indices,
            const sofa::type::vector< Topology::EdgeID >& inv_indices)
        : core::topology::TopologyChange(core::topology::EDGESRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    ~EdgesRenumbering() override;

    const sofa::type::vector<Topology::EdgeID> &getIndexArray() const { return indexArray; }

    const sofa::type::vector<Topology::EdgeID> &getinv_IndexArray() const { return inv_indexArray; }

    sofa::type::vector<Topology::EdgeID> indexArray;
    sofa::type::vector<Topology::EdgeID> inv_indexArray;
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Triangle Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two Triangles are being swapped */
class SOFA_CORE_API TrianglesIndicesSwap : public core::topology::TopologyChange
{
public:
    TrianglesIndicesSwap(const Topology::TriangleID i1,const Topology::TriangleID i2) : core::topology::TopologyChange(core::topology::TRIANGLESINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    ~TrianglesIndicesSwap() override;

    Topology::TriangleID index[2];
};


/** indicates that some triangles were added */
class SOFA_CORE_API TrianglesAdded : public core::topology::TopologyChange
{
public:
    TrianglesAdded(const size_t nT) : core::topology::TopologyChange(core::topology::TRIANGLESADDED),
        nTriangles(nT)
    { }

    TrianglesAdded(const size_t nT,
            const sofa::type::vector< Topology::Triangle >& _triangleArray,
            const sofa::type::vector< Topology::TriangleID >& trianglesIndex)
        : core::topology::TopologyChange(core::topology::TRIANGLESADDED),
          nTriangles(nT),
          triangleArray(_triangleArray),
          triangleIndexArray(trianglesIndex)
    { }

    TrianglesAdded(const size_t nT,
            const sofa::type::vector< Topology::Triangle >& _triangleArray,
            const sofa::type::vector< Topology::TriangleID >& trianglesIndex,
            const sofa::type::vector< sofa::type::vector< Topology::TriangleID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::TRIANGLESADDED),
          nTriangles(nT),
          triangleArray(_triangleArray),
          triangleIndexArray(trianglesIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }
    
    TrianglesAdded(const size_t nT,
            const sofa::type::vector< Topology::Triangle >& _triangleArray,
            const sofa::type::vector< Topology::TriangleID >& trianglesIndex,
            const sofa::type::vector< TriangleAncestorElem >& srcElems,
            const sofa::type::vector< sofa::type::vector< Topology::TriangleID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::TRIANGLESADDED),
          nTriangles(nT),
          triangleArray(_triangleArray),
          triangleIndexArray(trianglesIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
        , ancestorElems(srcElems)
    { }

    ~TrianglesAdded() override;

    size_t getNbAddedTriangles() const
    {
        return nTriangles;
    }

    const sofa::type::vector<Topology::TriangleID> &getArray() const
    {
        return triangleIndexArray;
    }

    const Topology::Triangle &getTriangle(const Topology::TriangleID i)
    {
        return triangleArray[i];
    }

    size_t getNbAddedElements() const { return nTriangles; }
    const sofa::type::vector< Topology::TriangleID >& getIndexArray() const { return triangleIndexArray; }
    const sofa::type::vector< Topology::Triangle >& getElementArray() const { return triangleArray; }

    size_t nTriangles;
    sofa::type::vector< Topology::Triangle > triangleArray;
    sofa::type::vector< Topology::TriangleID > triangleIndexArray;
    sofa::type::vector< sofa::type::vector< Topology::TriangleID > > ancestorsList;
    sofa::type::vector< sofa::type::vector< SReal > > coefs;
    sofa::type::vector< TriangleAncestorElem > ancestorElems;
};


/** indicates that some triangles are about to be removed */
class SOFA_CORE_API TrianglesRemoved : public core::topology::TopologyChange
{
public:
    TrianglesRemoved(const sofa::type::vector<Topology::TriangleID> _tArray) : core::topology::TopologyChange(core::topology::TRIANGLESREMOVED),
        removedTrianglesArray(_tArray)
    {}

    ~TrianglesRemoved() override;

    std::size_t getNbRemovedTriangles() const
    {
        return removedTrianglesArray.size();
    }

    const sofa::type::vector<Topology::TriangleID> &getArray() const
    {
        return removedTrianglesArray;
    }

    Topology::TriangleID &getTriangleIndices(const Topology::TriangleID i)
    {
        return removedTrianglesArray[i];
    }

protected:
    sofa::type::vector<Topology::TriangleID> removedTrianglesArray;
};


/** indicates that some triangles are about to be moved (i.e some/all of their vertices have just been moved)
 * TrianglesMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API TrianglesMoved_Removing : public core::topology::TopologyChange
{
public:
    TrianglesMoved_Removing (const sofa::type::vector< Topology::TriangleID >& triangleShell)
        : core::topology::TopologyChange (core::topology::TRIANGLESMOVED_REMOVING),
          trianglesAroundVertexMoved (triangleShell)
    {}

    ~TrianglesMoved_Removing() override;
    
    const sofa::type::vector< Topology::TriangleID >& getIndexArray() const { return trianglesAroundVertexMoved; }

    sofa::type::vector< Topology::TriangleID > trianglesAroundVertexMoved;
};


/** indicates that some triangles are about to be moved (i.e some/all of their vertices have just been moved)
 * TrianglesMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API TrianglesMoved_Adding : public core::topology::TopologyChange
{
public:
    TrianglesMoved_Adding (const sofa::type::vector< Topology::TriangleID >& triangleShell,
            const sofa::type::vector< Topology::Triangle >& triangleArray)
        : core::topology::TopologyChange (core::topology::TRIANGLESMOVED_ADDING),
          trianglesAroundVertexMoved (triangleShell), triangleArray2Moved (triangleArray)
    {}

    ~TrianglesMoved_Adding() override;
    
    const sofa::type::vector< Topology::TriangleID >& getIndexArray() const { return trianglesAroundVertexMoved; }
    const sofa::type::vector< Topology::Triangle >& getElementArray() const { return triangleArray2Moved; }

    sofa::type::vector< Topology::TriangleID > trianglesAroundVertexMoved;
    const sofa::type::vector< Topology::Triangle > triangleArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API TrianglesRenumbering : public core::topology::TopologyChange
{
public:

    TrianglesRenumbering() : core::topology::TopologyChange(core::topology::TRIANGLESRENUMBERING)
    { }

    TrianglesRenumbering(const sofa::type::vector< Topology::TriangleID >& indices,
            const sofa::type::vector< Topology::TriangleID >& inv_indices)
        : core::topology::TopologyChange(core::topology::TRIANGLESRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    ~TrianglesRenumbering() override;

    const sofa::type::vector<Topology::TriangleID> &getIndexArray() const { return indexArray; }

    const sofa::type::vector<Topology::TriangleID> &getinv_IndexArray() const { return inv_indexArray; }

    sofa::type::vector<Topology::TriangleID> indexArray;
    sofa::type::vector<Topology::TriangleID> inv_indexArray;
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Quad Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two Quads are being swapped */
class SOFA_CORE_API QuadsIndicesSwap : public core::topology::TopologyChange
{
public:
    QuadsIndicesSwap(const Topology::QuadID i1,const Topology::QuadID i2) : core::topology::TopologyChange(core::topology::QUADSINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    ~QuadsIndicesSwap() override;

    Topology::QuadID index[2];
};


/** indicates that some quads were added */
class SOFA_CORE_API QuadsAdded : public core::topology::TopologyChange
{
public:
    QuadsAdded(const size_t nT) : core::topology::TopologyChange(core::topology::QUADSADDED),
        nQuads(nT)
    { }

    QuadsAdded(const size_t nT,
            const sofa::type::vector< Topology::Quad >& _quadArray,
            const sofa::type::vector< Topology::QuadID >& quadsIndex)
        : core::topology::TopologyChange(core::topology::QUADSADDED),
          nQuads(nT),
          quadArray(_quadArray),
          quadIndexArray(quadsIndex)
    { }

    QuadsAdded(const size_t nT,
            const sofa::type::vector< Topology::Quad >& _quadArray,
            const sofa::type::vector< Topology::QuadID >& quadsIndex,
            const sofa::type::vector< sofa::type::vector< Topology::QuadID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::QUADSADDED),
          nQuads(nT),
          quadArray(_quadArray),
          quadIndexArray(quadsIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }
    
    QuadsAdded(const size_t nT,
            const sofa::type::vector< Topology::Quad >& _quadArray,
            const sofa::type::vector< Topology::QuadID >& quadsIndex,
            const sofa::type::vector< QuadAncestorElem >& srcElems,
            const sofa::type::vector< sofa::type::vector< Topology::QuadID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::QUADSADDED),
          nQuads(nT),
          quadArray(_quadArray),
          quadIndexArray(quadsIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs),
          ancestorElems(srcElems)
    { }

    ~QuadsAdded() override;

    size_t getNbAddedQuads() const
    {
        return nQuads;
    }

    const sofa::type::vector<Topology::QuadID> &getArray() const
    {
        return quadIndexArray;
    }

    const Topology::Quad &getQuad(const Topology::QuadID i) const
    {
        return quadArray[i];
    }

    size_t getNbAddedElements() const { return nQuads; }
    const sofa::type::vector< Topology::QuadID >& getIndexArray() const { return quadIndexArray; }
    const sofa::type::vector< Topology::Quad >& getElementArray() const { return quadArray; }

    size_t nQuads;
    sofa::type::vector< Topology::Quad > quadArray;
    sofa::type::vector< Topology::QuadID > quadIndexArray;
    sofa::type::vector< sofa::type::vector< Topology::QuadID > > ancestorsList;
    sofa::type::vector< sofa::type::vector< SReal > > coefs;
    sofa::type::vector< QuadAncestorElem > ancestorElems;
};


/** indicates that some quads are about to be removed */
class SOFA_CORE_API QuadsRemoved : public core::topology::TopologyChange
{
public:
    QuadsRemoved(const sofa::type::vector<Topology::QuadID> _qArray) : core::topology::TopologyChange(core::topology::QUADSREMOVED),
        removedQuadsArray(_qArray)
    { }

    ~QuadsRemoved() override;

    std::size_t getNbRemovedQuads() const
    {
        return removedQuadsArray.size();
    }

    const sofa::type::vector<Topology::QuadID> &getArray() const
    {
        return removedQuadsArray;
    }

    Topology::QuadID &getQuadIndices(const Topology::QuadID i)
    {
        return removedQuadsArray[i];
    }

protected:
    sofa::type::vector<Topology::QuadID> removedQuadsArray;
};


/** indicates that some quads are about to be moved (i.e some/all of their vertices have just been moved)
 * QuadsMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API QuadsMoved_Removing : public core::topology::TopologyChange
{
public:
    QuadsMoved_Removing (const sofa::type::vector< Topology::QuadID >& quadShell) : core::topology::TopologyChange (core::topology::QUADSMOVED_REMOVING),
        quadsAroundVertexMoved (quadShell)
    {}

    ~QuadsMoved_Removing() override;
    
    const sofa::type::vector< Topology::QuadID >& getIndexArray() const { return quadsAroundVertexMoved; }

    sofa::type::vector< Topology::QuadID > quadsAroundVertexMoved;
};


/** indicates that some quads are about to be moved (i.e some/all of their vertices have just been moved)
 * QuadsMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API QuadsMoved_Adding : public core::topology::TopologyChange
{
public:
    QuadsMoved_Adding (const sofa::type::vector< Topology::QuadID >& quadShell,
            const sofa::type::vector< Topology::Quad >& quadArray)
        : core::topology::TopologyChange (core::topology::QUADSMOVED_ADDING),
          quadsAroundVertexMoved (quadShell), quadArray2Moved (quadArray)
    {}

    ~QuadsMoved_Adding() override;
    
    const sofa::type::vector< Topology::QuadID >& getIndexArray() const { return quadsAroundVertexMoved; }
    const sofa::type::vector< Topology::Quad >& getElementArray() const { return quadArray2Moved; }

    sofa::type::vector< Topology::QuadID > quadsAroundVertexMoved;
    const sofa::type::vector< Topology::Quad > quadArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API QuadsRenumbering : public core::topology::TopologyChange
{
public:

    QuadsRenumbering() : core::topology::TopologyChange(core::topology::QUADSRENUMBERING)
    { }

    ~QuadsRenumbering() override;

    QuadsRenumbering(const sofa::type::vector< Topology::QuadID >& indices,
            const sofa::type::vector< Topology::QuadID >& inv_indices)
        : core::topology::TopologyChange(core::topology::QUADSRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    const sofa::type::vector<Topology::QuadID> &getIndexArray() const { return indexArray; }

    const sofa::type::vector<Topology::QuadID> &getinv_IndexArray() const { return inv_indexArray; }

    sofa::type::vector<Topology::QuadID> indexArray;
    sofa::type::vector<Topology::QuadID> inv_indexArray;
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Tetrahedron Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two Tetrahedra are being swapped */
class SOFA_CORE_API TetrahedraIndicesSwap : public core::topology::TopologyChange
{
public:
    TetrahedraIndicesSwap(const Topology::TetrahedronID i1,const Topology::TetrahedronID i2) : core::topology::TopologyChange(core::topology::TETRAHEDRAINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    ~TetrahedraIndicesSwap() override;

    Topology::TetrahedronID index[2];
};


/** indicates that some tetrahedra were added */
class SOFA_CORE_API TetrahedraAdded : public core::topology::TopologyChange
{
public:
    TetrahedraAdded(const size_t nT) : core::topology::TopologyChange(core::topology::TETRAHEDRAADDED),
        nTetrahedra(nT)
    { }

    TetrahedraAdded(const size_t nT,
            const sofa::type::vector< Topology::Tetrahedron >& _tetrahedronArray,
            const sofa::type::vector< Topology::TetrahedronID >& tetrahedraIndex)
        : core::topology::TopologyChange(core::topology::TETRAHEDRAADDED),
          nTetrahedra(nT),
          tetrahedronArray(_tetrahedronArray),
          tetrahedronIndexArray(tetrahedraIndex)
    { }

    TetrahedraAdded(const size_t nT,
            const sofa::type::vector< Topology::Tetrahedron >& _tetrahedronArray,
            const sofa::type::vector< Topology::TetrahedronID >& tetrahedraIndex,
            const sofa::type::vector< sofa::type::vector< Topology::TetrahedronID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::TETRAHEDRAADDED),
          nTetrahedra(nT),
          tetrahedronArray(_tetrahedronArray),
          tetrahedronIndexArray(tetrahedraIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }
    
    TetrahedraAdded(const size_t nT,
            const sofa::type::vector< Topology::Tetrahedron >& _tetrahedronArray,
            const sofa::type::vector< Topology::TetrahedronID >& tetrahedraIndex,
            const sofa::type::vector< TetrahedronAncestorElem >& srcElems,
            const sofa::type::vector< sofa::type::vector< Topology::TetrahedronID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::TETRAHEDRAADDED),
          nTetrahedra(nT),
          tetrahedronArray(_tetrahedronArray),
          tetrahedronIndexArray(tetrahedraIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs),
          ancestorElems(srcElems)
    { }

    ~TetrahedraAdded() override;

    const sofa::type::vector<Topology::TetrahedronID> &getArray() const
    {
        return tetrahedronIndexArray;
    }

    size_t getNbAddedTetrahedra() const
    {
        return nTetrahedra;
    }

    size_t getNbAddedElements() const { return nTetrahedra; }
    const sofa::type::vector< Topology::TetrahedronID >& getIndexArray() const { return tetrahedronIndexArray; }
    const sofa::type::vector< Topology::Tetrahedron >& getElementArray() const { return tetrahedronArray; }

    size_t nTetrahedra;
    sofa::type::vector< Topology::Tetrahedron > tetrahedronArray;
    sofa::type::vector< Topology::TetrahedronID > tetrahedronIndexArray;
    sofa::type::vector< sofa::type::vector< Topology::TetrahedronID > > ancestorsList;
    sofa::type::vector< sofa::type::vector< SReal > > coefs;
    sofa::type::vector< TetrahedronAncestorElem > ancestorElems;
};

/** indicates that some tetrahedra are about to be removed */
class SOFA_CORE_API TetrahedraRemoved : public core::topology::TopologyChange
{
public:
    TetrahedraRemoved(const sofa::type::vector<Topology::TetrahedronID> _tArray)
        : core::topology::TopologyChange(core::topology::TETRAHEDRAREMOVED),
          removedTetrahedraArray(_tArray)
    { }

    ~TetrahedraRemoved() override;

    const sofa::type::vector<Topology::TetrahedronID> &getArray() const
    {
        return removedTetrahedraArray;
    }

    std::size_t getNbRemovedTetrahedra() const
    {
        return removedTetrahedraArray.size();
    }

    sofa::type::vector<Topology::TetrahedronID> removedTetrahedraArray;
};


/** indicates that some tetrahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * TetrahedraMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API TetrahedraMoved_Removing : public core::topology::TopologyChange
{
public:
    TetrahedraMoved_Removing (const sofa::type::vector< Topology::TetrahedronID >& tetrahedronShell)
        : core::topology::TopologyChange (core::topology::TETRAHEDRAMOVED_REMOVING),
          tetrahedraAroundVertexMoved (tetrahedronShell)
    {}

    ~TetrahedraMoved_Removing() override;
    
    const sofa::type::vector< Topology::TetrahedronID >& getIndexArray() const { return tetrahedraAroundVertexMoved; }

    sofa::type::vector< Topology::TetrahedronID > tetrahedraAroundVertexMoved;
};


/** indicates that some tetrahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * TetrahedraMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API TetrahedraMoved_Adding : public core::topology::TopologyChange
{
public:
    TetrahedraMoved_Adding (const sofa::type::vector< Topology::TetrahedronID >& tetrahedronShell,
            const sofa::type::vector< Topology::Tetrahedron >& tetrahedronArray)
        : core::topology::TopologyChange (core::topology::TETRAHEDRAMOVED_ADDING),
          tetrahedraAroundVertexMoved (tetrahedronShell), tetrahedronArray2Moved (tetrahedronArray)
    {}

    ~TetrahedraMoved_Adding() override;
    
    const sofa::type::vector< Topology::TetrahedronID >& getIndexArray() const { return tetrahedraAroundVertexMoved; }
    const sofa::type::vector< Topology::Tetrahedron >& getElementArray() const { return tetrahedronArray2Moved; }

    sofa::type::vector< Topology::TetrahedronID > tetrahedraAroundVertexMoved;
    const sofa::type::vector< Topology::Tetrahedron > tetrahedronArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API TetrahedraRenumbering : public core::topology::TopologyChange
{
public:

    TetrahedraRenumbering()
        : core::topology::TopologyChange(core::topology::TETRAHEDRARENUMBERING)
    { }

    TetrahedraRenumbering(const sofa::type::vector< Topology::TetrahedronID >& indices,
            const sofa::type::vector< Topology::TetrahedronID >& inv_indices)
        : core::topology::TopologyChange(core::topology::TETRAHEDRARENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    ~TetrahedraRenumbering() override;

    const sofa::type::vector<Topology::TetrahedronID> &getIndexArray() const { return indexArray; }

    const sofa::type::vector<Topology::TetrahedronID> &getinv_IndexArray() const { return inv_indexArray; }

    sofa::type::vector<Topology::TetrahedronID> indexArray;
    sofa::type::vector<Topology::TetrahedronID> inv_indexArray;
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Hexahedron Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two Hexahedra are being swapped */
class SOFA_CORE_API HexahedraIndicesSwap : public core::topology::TopologyChange
{
public:
    HexahedraIndicesSwap(const Topology::HexahedronID i1,const Topology::HexahedronID i2) : core::topology::TopologyChange(core::topology::HEXAHEDRAINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    ~HexahedraIndicesSwap() override;

    Topology::HexahedronID index[2];
};


/** indicates that some hexahedra were added */
class SOFA_CORE_API HexahedraAdded : public core::topology::TopologyChange
{
public:
    HexahedraAdded(const size_t nT) : core::topology::TopologyChange(core::topology::HEXAHEDRAADDED),
        nHexahedra(nT)
    { }

    HexahedraAdded(const size_t nT,
            const sofa::type::vector< Topology::Hexahedron >& _hexahedronArray,
            const sofa::type::vector< Topology::HexahedronID >& hexahedraIndex)
        : core::topology::TopologyChange(core::topology::HEXAHEDRAADDED),
          nHexahedra(nT),
          hexahedronArray(_hexahedronArray),
          hexahedronIndexArray(hexahedraIndex)
    { }

    HexahedraAdded(const size_t nT,
            const sofa::type::vector< Topology::Hexahedron >& _hexahedronArray,
            const sofa::type::vector< Topology::HexahedronID >& hexahedraIndex,
            const sofa::type::vector< sofa::type::vector< Topology::HexahedronID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::HEXAHEDRAADDED),
          nHexahedra(nT),
          hexahedronArray(_hexahedronArray),
          hexahedronIndexArray(hexahedraIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }
    
    HexahedraAdded(const size_t nT,
            const sofa::type::vector< Topology::Hexahedron >& _hexahedronArray,
            const sofa::type::vector< Topology::HexahedronID >& hexahedraIndex,
            const sofa::type::vector< HexahedronAncestorElem >& srcElems,
            const sofa::type::vector< sofa::type::vector< Topology::HexahedronID > >& ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::HEXAHEDRAADDED),
          nHexahedra(nT),
          hexahedronArray(_hexahedronArray),
          hexahedronIndexArray(hexahedraIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs),
          ancestorElems(srcElems)
    { }

    ~HexahedraAdded() override;

    size_t getNbAddedHexahedra() const
    {
        return nHexahedra;
    }

    size_t getNbAddedElements() const { return nHexahedra; }
    const sofa::type::vector< Topology::HexahedronID >& getIndexArray() const { return hexahedronIndexArray; }
    const sofa::type::vector< Topology::Hexahedron >& getElementArray() const { return hexahedronArray; }

    size_t nHexahedra;
    sofa::type::vector< Topology::Hexahedron > hexahedronArray;
    sofa::type::vector< Topology::HexahedronID > hexahedronIndexArray;
    sofa::type::vector< sofa::type::vector< Topology::HexahedronID > > ancestorsList;
    sofa::type::vector< sofa::type::vector< SReal > > coefs;
    sofa::type::vector< HexahedronAncestorElem > ancestorElems;
};

/** indicates that some hexahedra are about to be removed */
class SOFA_CORE_API HexahedraRemoved : public core::topology::TopologyChange
{
public:
    HexahedraRemoved(const sofa::type::vector<Topology::HexahedronID> _tArray)
        : core::topology::TopologyChange(core::topology::HEXAHEDRAREMOVED),
          removedHexahedraArray(_tArray)
    { }

    ~HexahedraRemoved() override;

    const sofa::type::vector<Topology::HexahedronID> &getArray() const
    {
        return removedHexahedraArray;
    }

    std::size_t getNbRemovedHexahedra() const
    {
        return removedHexahedraArray.size();
    }

    sofa::type::vector<Topology::HexahedronID> removedHexahedraArray;
};


/** indicates that some hexahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * HexahedraMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API HexahedraMoved_Removing : public core::topology::TopologyChange
{
public:
    HexahedraMoved_Removing (const sofa::type::vector< Topology::HexahedronID >& hexahedronShell)
        : core::topology::TopologyChange (core::topology::HEXAHEDRAMOVED_REMOVING),
          hexahedraAroundVertexMoved (hexahedronShell)
    {}

    ~HexahedraMoved_Removing() override;

    const sofa::type::vector< Topology::HexahedronID >& getIndexArray() const { return hexahedraAroundVertexMoved; }

    sofa::type::vector< Topology::HexahedronID > hexahedraAroundVertexMoved;
};


/** indicates that some hexahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * HexahedraMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API HexahedraMoved_Adding : public core::topology::TopologyChange
{
public:
    HexahedraMoved_Adding (const sofa::type::vector< Topology::HexahedronID >& hexahedronShell,
            const sofa::type::vector< Topology::Hexahedron >& hexahedronArray)
        : core::topology::TopologyChange (core::topology::HEXAHEDRAMOVED_ADDING),
          hexahedraAroundVertexMoved (hexahedronShell), hexahedronArray2Moved (hexahedronArray)
    {}

    ~HexahedraMoved_Adding() override;
    
    const sofa::type::vector< Topology::HexahedronID >& getIndexArray() const { return hexahedraAroundVertexMoved; }
    const sofa::type::vector< Topology::Hexahedron >& getElementArray() const { return hexahedronArray2Moved; }

    sofa::type::vector< Topology::HexahedronID > hexahedraAroundVertexMoved;
    const sofa::type::vector< Topology::Hexahedron > hexahedronArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API HexahedraRenumbering : public core::topology::TopologyChange
{
public:

    HexahedraRenumbering() : core::topology::TopologyChange(core::topology::HEXAHEDRARENUMBERING)
    { }

    HexahedraRenumbering(const sofa::type::vector< Topology::HexahedronID >& indices,
            const sofa::type::vector< Topology::HexahedronID >& inv_indices)
        : core::topology::TopologyChange(core::topology::HEXAHEDRARENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    ~HexahedraRenumbering() override;

    const sofa::type::vector<Topology::HexahedronID> &getIndexArray() const { return indexArray; }

    const sofa::type::vector<Topology::HexahedronID> &getinv_IndexArray() const { return inv_indexArray; }

    sofa::type::vector<Topology::HexahedronID> indexArray;
    sofa::type::vector<Topology::HexahedronID> inv_indexArray;
};
} // namespace sofa::core::topology

#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGYCHANGE_DEFINITION
namespace std
{
    extern template class SOFA_CORE_API std::list<const sofa::core::topology::TopologyChange*>;
}
namespace sofa::core::objectmodel
{
    extern template class SOFA_CORE_API Data<std::list<const sofa::core::topology::TopologyChange*>>;
}

#endif /// SOFA_CORE_TOPOLOGY_BASETOPOLOGYENGINE_DEFINITION
