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
#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGYCHANGE_H
#define SOFA_CORE_TOPOLOGY_TOPOLOGYCHANGE_H

#include <sofa/core/topology/Topology.h>
#include <iostream>

namespace sofa
{

namespace core
{

namespace topology
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
    TopologyElemID() : type(POINT), index((unsigned int)-1) {}

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

    PointAncestorElem() : type(POINT), index((unsigned int)-1) {}

    PointAncestorElem(TopologyObjectType _type, unsigned int _index, const LocalCoords& _localCoords)
        : type(_type)
        , index(_index)
        , localCoords(_localCoords)
    {}
    
    TopologyObjectType type;
    unsigned int index;
    LocalCoords localCoords;
};

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const PointAncestorElem& d);
SOFA_CORE_API std::istream& operator >> (std::istream& in, PointAncestorElem& d);

/// Topology change informations related to the ancestor topology element of an edge
template<int NV>
struct ElemAncestorElem
{

    ElemAncestorElem()
    {}

    ElemAncestorElem(const helper::fixed_array<PointAncestorElem,NV>& _pointSrcElems,
        const helper::vector<TopologyElemID>& _srcElems)
        : pointSrcElems(_pointSrcElems)
        , srcElems(_srcElems)
    {}
    
    ElemAncestorElem(const helper::fixed_array<PointAncestorElem,NV>& _pointSrcElems,
        const TopologyElemID& _srcElem)
        : pointSrcElems(_pointSrcElems)
        , srcElems()
    {
        srcElems.push_back(_srcElem);
    }
    
    helper::fixed_array<PointAncestorElem,NV> pointSrcElems;
    helper::vector<TopologyElemID> srcElems;
};

template<int NV>
SOFA_CORE_API std::ostream& operator << (std::ostream& out, const ElemAncestorElem<NV>& d);
template<int NV>
SOFA_CORE_API std::istream& operator >> (std::istream& in, ElemAncestorElem<NV>& d);

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
    enum { USE_EMOVED_ADDIpNG   = 1 };

    typedef EdgesIndicesSwap    EIndicesSwap;
    typedef EdgesRenumbering    ERenumbering;
    typedef EdgesAdded          EAdded;
    typedef EdgesRemoved        ERemoved;
    typedef EdgesMoved_Removing EMoved_Removing;
    typedef EdgesMoved_Adding   EMoved_Adding;
    /// This event is not used for this type of element
    class EMoved { };

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
class SOFA_CORE_API EndingEvent : public core::topology::TopologyChange
{
public:
    EndingEvent()
        : core::topology::TopologyChange(core::topology::ENDING_EVENT)
    {}

    virtual ~EndingEvent();
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   Point Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two points are being swapped */
class SOFA_CORE_API PointsIndicesSwap : public core::topology::TopologyChange
{
public:
    PointsIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::POINTSINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    virtual ~PointsIndicesSwap();

public:
    unsigned int index[2];
};


/** indicates that some points were added */
class SOFA_CORE_API PointsAdded : public core::topology::TopologyChange
{
public:

    PointsAdded(const unsigned int nV) : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV)
    { }

    PointsAdded(const unsigned int nV,
            const sofa::helper::vector< unsigned int >& indices)
        : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV), pointIndexArray(indices)
    { }

    PointsAdded(const unsigned int nV,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double       > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV), ancestorsList(ancestors), coefs(baryCoefs)
    { }

    PointsAdded(const unsigned int nV,
            const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double       > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV), pointIndexArray(indices), ancestorsList(ancestors), coefs(baryCoefs)
    { }

    PointsAdded(const unsigned int nV,
            const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< PointAncestorElem >& srcElems,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double       > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::POINTSADDED)
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
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return pointIndexArray; }
    const sofa::helper::vector< Topology::Point >& getElementArray() const { return pointIndexArray; }


public:
    unsigned int nVertices;
    sofa::helper::vector< unsigned int > pointIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double       > > coefs;
    sofa::helper::vector< PointAncestorElem > ancestorElems;
};


/** indicates that some points are about to be removed */
class SOFA_CORE_API PointsRemoved : public core::topology::TopologyChange
{
public:
    PointsRemoved(const sofa::helper::vector<unsigned int>& _vArray) : core::topology::TopologyChange(core::topology::POINTSREMOVED),
        removedVertexArray(_vArray)
    { }

    virtual ~PointsRemoved();

    const sofa::helper::vector<unsigned int> &getArray() const { return removedVertexArray;	}

public:
    sofa::helper::vector<unsigned int> removedVertexArray;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API PointsRenumbering : public core::topology::TopologyChange
{
public:

    PointsRenumbering() : core::topology::TopologyChange(core::topology::POINTSRENUMBERING)
    { }

    PointsRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::topology::TopologyChange(core::topology::POINTSRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    virtual ~PointsRenumbering();

    const sofa::helper::vector<unsigned int> &getIndexArray() const { return indexArray; }

    const sofa::helper::vector<unsigned int> &getinv_IndexArray() const { return inv_indexArray; }

public:
    sofa::helper::vector<unsigned int> indexArray;
    sofa::helper::vector<unsigned int> inv_indexArray;
};


/** indicates that some points were moved */
class SOFA_CORE_API PointsMoved : public core::topology::TopologyChange
{
public:

    PointsMoved(const sofa::helper::vector<unsigned int>& indices,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::POINTSMOVED)
        , indicesList(indices), ancestorsList(ancestors), baryCoefsList(baryCoefs)
    {}

    virtual ~PointsMoved();
    
    const sofa::helper::vector<unsigned int> &getIndexArray() const { return indicesList; }

public:
    sofa::helper::vector<unsigned int> indicesList;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > baryCoefsList;
};





////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Edge Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two edges are being swapped */
class SOFA_CORE_API EdgesIndicesSwap : public core::topology::TopologyChange
{
public:
    EdgesIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::EDGESINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    virtual ~EdgesIndicesSwap();

public:
    unsigned int index[2];
};


/** indicates that some edges were added */
class SOFA_CORE_API EdgesAdded : public core::topology::TopologyChange
{
public:
    EdgesAdded(const unsigned int nE) : core::topology::TopologyChange(core::topology::EDGESADDED),
        nEdges(nE)
    { }

    EdgesAdded(const unsigned int nE,
            const sofa::helper::vector< Topology::Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndex)
        : core::topology::TopologyChange(core::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex)
    { }

    EdgesAdded(const unsigned int nE,
            const sofa::helper::vector< Topology::Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndex,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors)
        : core::topology::TopologyChange(core::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex),
          ancestorsList(ancestors)
    { }

    EdgesAdded(const unsigned int nE,
            const sofa::helper::vector< Topology::Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndex,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }
    
    EdgesAdded(const unsigned int nE,
            const sofa::helper::vector< Topology::Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndex,
            const sofa::helper::vector< EdgeAncestorElem >& srcElems,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex),
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
    const sofa::helper::vector< Topology::Edge > &getArray() const
    {
        return edgeArray;
    }

    unsigned int getNbAddedElements() const { return nEdges; }
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return edgeIndexArray; }
    const sofa::helper::vector< Topology::Edge >& getElementArray() const { return edgeArray; }

public:
    unsigned int nEdges;
    sofa::helper::vector< Topology::Edge > edgeArray;
    sofa::helper::vector< unsigned int > edgeIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
    sofa::helper::vector< EdgeAncestorElem > ancestorElems;
};


/** indicates that some edges are about to be removed */
class SOFA_CORE_API EdgesRemoved : public core::topology::TopologyChange
{
public:
    EdgesRemoved(const sofa::helper::vector<unsigned int> _eArray) : core::topology::TopologyChange(core::topology::EDGESREMOVED),
        removedEdgesArray(_eArray)
    {}

    virtual ~EdgesRemoved();

    virtual const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedEdgesArray;
    }

    virtual std::size_t getNbRemovedEdges() const
    {
        return removedEdgesArray.size();
    }

public:
    sofa::helper::vector<unsigned int> removedEdgesArray;
};


/** indicates that some edges are about to be moved (i.e one or both of their vertices have just been moved)
 * EdgesMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API EdgesMoved_Removing : public core::topology::TopologyChange
{
public:
    EdgesMoved_Removing (const sofa::helper::vector< unsigned int >& edgeShell) : core::topology::TopologyChange (core::topology::EDGESMOVED_REMOVING),
        edgesAroundVertexMoved (edgeShell)
    {}

    virtual ~EdgesMoved_Removing();
    
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return edgesAroundVertexMoved; }

public:
    sofa::helper::vector< unsigned int > edgesAroundVertexMoved;
};


/** indicates that some edges are about to be moved (i.e one or both of their vertices have just been moved)
 * EdgesMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API EdgesMoved_Adding : public core::topology::TopologyChange
{
public:
    EdgesMoved_Adding (const sofa::helper::vector< unsigned int >& edgeShell,
            const sofa::helper::vector< Topology::Edge >& edgeArray)
        : core::topology::TopologyChange (core::topology::EDGESMOVED_ADDING),
          edgesAroundVertexMoved (edgeShell), edgeArray2Moved (edgeArray)
    {}

    virtual ~EdgesMoved_Adding();
    
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return edgesAroundVertexMoved; }
    const sofa::helper::vector< Topology::Edge >& getElementArray() const { return edgeArray2Moved; }

public:
    sofa::helper::vector< unsigned int > edgesAroundVertexMoved;
    sofa::helper::vector< Topology::Edge > edgeArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API EdgesRenumbering : public core::topology::TopologyChange
{
public:
    EdgesRenumbering() : core::topology::TopologyChange(core::topology::EDGESRENUMBERING)
    { }

    EdgesRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::topology::TopologyChange(core::topology::EDGESRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    virtual ~EdgesRenumbering();

    const sofa::helper::vector<unsigned int> &getIndexArray() const { return indexArray; }

    const sofa::helper::vector<unsigned int> &getinv_IndexArray() const { return inv_indexArray; }

public:
    sofa::helper::vector<unsigned int> indexArray;
    sofa::helper::vector<unsigned int> inv_indexArray;
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Triangle Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two Triangles are being swapped */
class SOFA_CORE_API TrianglesIndicesSwap : public core::topology::TopologyChange
{
public:
    TrianglesIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::TRIANGLESINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    virtual ~TrianglesIndicesSwap();

public:
    unsigned int index[2];
};


/** indicates that some triangles were added */
class SOFA_CORE_API TrianglesAdded : public core::topology::TopologyChange
{
public:
    TrianglesAdded(const unsigned int nT) : core::topology::TopologyChange(core::topology::TRIANGLESADDED),
        nTriangles(nT)
    { }

    TrianglesAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Triangle >& _triangleArray,
            const sofa::helper::vector< unsigned int >& trianglesIndex)
        : core::topology::TopologyChange(core::topology::TRIANGLESADDED),
          nTriangles(nT),
          triangleArray(_triangleArray),
          triangleIndexArray(trianglesIndex)
    { }

    TrianglesAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Triangle >& _triangleArray,
            const sofa::helper::vector< unsigned int >& trianglesIndex,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::TRIANGLESADDED),
          nTriangles(nT),
          triangleArray(_triangleArray),
          triangleIndexArray(trianglesIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }
    
    TrianglesAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Triangle >& _triangleArray,
            const sofa::helper::vector< unsigned int >& trianglesIndex,
            const sofa::helper::vector< TriangleAncestorElem >& srcElems,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::TRIANGLESADDED),
          nTriangles(nT),
          triangleArray(_triangleArray),
          triangleIndexArray(trianglesIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
        , ancestorElems(srcElems)
    { }

    virtual ~TrianglesAdded();

    unsigned int getNbAddedTriangles() const
    {
        return nTriangles;
    }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return triangleIndexArray;
    }

    const Topology::Triangle &getTriangle(const unsigned int i)
    {
        return triangleArray[i];
    }

    unsigned int getNbAddedElements() const { return nTriangles; }
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return triangleIndexArray; }
    const sofa::helper::vector< Topology::Triangle >& getElementArray() const { return triangleArray; }

public:
    unsigned int nTriangles;
    sofa::helper::vector< Topology::Triangle > triangleArray;
    sofa::helper::vector< unsigned int > triangleIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
    sofa::helper::vector< TriangleAncestorElem > ancestorElems;
};


/** indicates that some triangles are about to be removed */
class SOFA_CORE_API TrianglesRemoved : public core::topology::TopologyChange
{
public:
    TrianglesRemoved(const sofa::helper::vector<unsigned int> _tArray) : core::topology::TopologyChange(core::topology::TRIANGLESREMOVED),
        removedTrianglesArray(_tArray)
    {}

    virtual ~TrianglesRemoved();

    std::size_t getNbRemovedTriangles() const
    {
        return removedTrianglesArray.size();
    }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedTrianglesArray;
    }

    unsigned int &getTriangleIndices(const unsigned int i)
    {
        return removedTrianglesArray[i];
    }

protected:
    sofa::helper::vector<unsigned int> removedTrianglesArray;
};


/** indicates that some triangles are about to be moved (i.e some/all of their vertices have just been moved)
 * TrianglesMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API TrianglesMoved_Removing : public core::topology::TopologyChange
{
public:
    TrianglesMoved_Removing (const sofa::helper::vector< unsigned int >& triangleShell)
        : core::topology::TopologyChange (core::topology::TRIANGLESMOVED_REMOVING),
          trianglesAroundVertexMoved (triangleShell)
    {}

    virtual ~TrianglesMoved_Removing();
    
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return trianglesAroundVertexMoved; }

public:
    sofa::helper::vector< unsigned int > trianglesAroundVertexMoved;
};


/** indicates that some triangles are about to be moved (i.e some/all of their vertices have just been moved)
 * TrianglesMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API TrianglesMoved_Adding : public core::topology::TopologyChange
{
public:
    TrianglesMoved_Adding (const sofa::helper::vector< unsigned int >& triangleShell,
            const sofa::helper::vector< Topology::Triangle >& triangleArray)
        : core::topology::TopologyChange (core::topology::TRIANGLESMOVED_ADDING),
          trianglesAroundVertexMoved (triangleShell), triangleArray2Moved (triangleArray)
    {}

    virtual ~TrianglesMoved_Adding();
    
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return trianglesAroundVertexMoved; }
    const sofa::helper::vector< Topology::Triangle >& getElementArray() const { return triangleArray2Moved; }

public:
    sofa::helper::vector< unsigned int > trianglesAroundVertexMoved;
    const sofa::helper::vector< Topology::Triangle > triangleArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API TrianglesRenumbering : public core::topology::TopologyChange
{
public:

    TrianglesRenumbering() : core::topology::TopologyChange(core::topology::TRIANGLESRENUMBERING)
    { }

    TrianglesRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::topology::TopologyChange(core::topology::TRIANGLESRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    virtual ~TrianglesRenumbering();

    const sofa::helper::vector<unsigned int> &getIndexArray() const { return indexArray; }

    const sofa::helper::vector<unsigned int> &getinv_IndexArray() const { return inv_indexArray; }

public:
    sofa::helper::vector<unsigned int> indexArray;
    sofa::helper::vector<unsigned int> inv_indexArray;
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Quad Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two Quads are being swapped */
class SOFA_CORE_API QuadsIndicesSwap : public core::topology::TopologyChange
{
public:
    QuadsIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::QUADSINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    virtual ~QuadsIndicesSwap();

public:
    unsigned int index[2];
};


/** indicates that some quads were added */
class SOFA_CORE_API QuadsAdded : public core::topology::TopologyChange
{
public:
    QuadsAdded(const unsigned int nT) : core::topology::TopologyChange(core::topology::QUADSADDED),
        nQuads(nT)
    { }

    QuadsAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Quad >& _quadArray,
            const sofa::helper::vector< unsigned int >& quadsIndex)
        : core::topology::TopologyChange(core::topology::QUADSADDED),
          nQuads(nT),
          quadArray(_quadArray),
          quadIndexArray(quadsIndex)
    { }

    QuadsAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Quad >& _quadArray,
            const sofa::helper::vector< unsigned int >& quadsIndex,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::QUADSADDED),
          nQuads(nT),
          quadArray(_quadArray),
          quadIndexArray(quadsIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }
    
    QuadsAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Quad >& _quadArray,
            const sofa::helper::vector< unsigned int >& quadsIndex,
            const sofa::helper::vector< QuadAncestorElem >& srcElems,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::QUADSADDED),
          nQuads(nT),
          quadArray(_quadArray),
          quadIndexArray(quadsIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs),
          ancestorElems(srcElems)
    { }

    virtual ~QuadsAdded();

    unsigned int getNbAddedQuads() const
    {
        return nQuads;
    }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return quadIndexArray;
    }

    const Topology::Quad &getQuad(const unsigned int i) const
    {
        return quadArray[i];
    }

    unsigned int getNbAddedElements() const { return nQuads; }
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return quadIndexArray; }
    const sofa::helper::vector< Topology::Quad >& getElementArray() const { return quadArray; }

public:
    unsigned int nQuads;
    sofa::helper::vector< Topology::Quad > quadArray;
    sofa::helper::vector< unsigned int > quadIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
    sofa::helper::vector< QuadAncestorElem > ancestorElems;
};


/** indicates that some quads are about to be removed */
class SOFA_CORE_API QuadsRemoved : public core::topology::TopologyChange
{
public:
    QuadsRemoved(const sofa::helper::vector<unsigned int> _qArray) : core::topology::TopologyChange(core::topology::QUADSREMOVED),
        removedQuadsArray(_qArray)
    { }

    virtual ~QuadsRemoved();

    std::size_t getNbRemovedQuads() const
    {
        return removedQuadsArray.size();
    }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedQuadsArray;
    }

    unsigned int &getQuadIndices(const unsigned int i)
    {
        return removedQuadsArray[i];
    }

protected:
    sofa::helper::vector<unsigned int> removedQuadsArray;
};


/** indicates that some quads are about to be moved (i.e some/all of their vertices have just been moved)
 * QuadsMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API QuadsMoved_Removing : public core::topology::TopologyChange
{
public:
    QuadsMoved_Removing (const sofa::helper::vector< unsigned int >& quadShell) : core::topology::TopologyChange (core::topology::QUADSMOVED_REMOVING),
        quadsAroundVertexMoved (quadShell)
    {}

    virtual ~QuadsMoved_Removing();
    
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return quadsAroundVertexMoved; }

public:
    sofa::helper::vector< unsigned int > quadsAroundVertexMoved;
};


/** indicates that some quads are about to be moved (i.e some/all of their vertices have just been moved)
 * QuadsMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API QuadsMoved_Adding : public core::topology::TopologyChange
{
public:
    QuadsMoved_Adding (const sofa::helper::vector< unsigned int >& quadShell,
            const sofa::helper::vector< Topology::Quad >& quadArray)
        : core::topology::TopologyChange (core::topology::QUADSMOVED_ADDING),
          quadsAroundVertexMoved (quadShell), quadArray2Moved (quadArray)
    {}

    virtual ~QuadsMoved_Adding();
    
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return quadsAroundVertexMoved; }
    const sofa::helper::vector< Topology::Quad >& getElementArray() const { return quadArray2Moved; }

public:
    sofa::helper::vector< unsigned int > quadsAroundVertexMoved;
    const sofa::helper::vector< Topology::Quad > quadArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API QuadsRenumbering : public core::topology::TopologyChange
{
public:

    QuadsRenumbering() : core::topology::TopologyChange(core::topology::QUADSRENUMBERING)
    { }

    virtual ~QuadsRenumbering();

    QuadsRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::topology::TopologyChange(core::topology::QUADSRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    const sofa::helper::vector<unsigned int> &getIndexArray() const { return indexArray; }

    const sofa::helper::vector<unsigned int> &getinv_IndexArray() const { return inv_indexArray; }

public:
    sofa::helper::vector<unsigned int> indexArray;
    sofa::helper::vector<unsigned int> inv_indexArray;
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Tetrahedron Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two Tetrahedra are being swapped */
class SOFA_CORE_API TetrahedraIndicesSwap : public core::topology::TopologyChange
{
public:
    TetrahedraIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::TETRAHEDRAINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    virtual ~TetrahedraIndicesSwap();

public:
    unsigned int index[2];
};


/** indicates that some tetrahedra were added */
class SOFA_CORE_API TetrahedraAdded : public core::topology::TopologyChange
{
public:
    TetrahedraAdded(const unsigned int nT) : core::topology::TopologyChange(core::topology::TETRAHEDRAADDED),
        nTetrahedra(nT)
    { }

    TetrahedraAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Tetrahedron >& _tetrahedronArray,
            const sofa::helper::vector< unsigned int >& tetrahedraIndex)
        : core::topology::TopologyChange(core::topology::TETRAHEDRAADDED),
          nTetrahedra(nT),
          tetrahedronArray(_tetrahedronArray),
          tetrahedronIndexArray(tetrahedraIndex)
    { }

    TetrahedraAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Tetrahedron >& _tetrahedronArray,
            const sofa::helper::vector< unsigned int >& tetrahedraIndex,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::TETRAHEDRAADDED),
          nTetrahedra(nT),
          tetrahedronArray(_tetrahedronArray),
          tetrahedronIndexArray(tetrahedraIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }
    
    TetrahedraAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Tetrahedron >& _tetrahedronArray,
            const sofa::helper::vector< unsigned int >& tetrahedraIndex,
            const sofa::helper::vector< TetrahedronAncestorElem >& srcElems,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::TETRAHEDRAADDED),
          nTetrahedra(nT),
          tetrahedronArray(_tetrahedronArray),
          tetrahedronIndexArray(tetrahedraIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs),
          ancestorElems(srcElems)
    { }

    virtual ~TetrahedraAdded();

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return tetrahedronIndexArray;
    }

    unsigned int getNbAddedTetrahedra() const
    {
        return nTetrahedra;
    }

    unsigned int getNbAddedElements() const { return nTetrahedra; }
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return tetrahedronIndexArray; }
    const sofa::helper::vector< Topology::Tetrahedron >& getElementArray() const { return tetrahedronArray; }

public:
    unsigned int nTetrahedra;
    sofa::helper::vector< Topology::Tetrahedron > tetrahedronArray;
    sofa::helper::vector< unsigned int > tetrahedronIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
    sofa::helper::vector< TetrahedronAncestorElem > ancestorElems;
};

/** indicates that some tetrahedra are about to be removed */
class SOFA_CORE_API TetrahedraRemoved : public core::topology::TopologyChange
{
public:
    TetrahedraRemoved(const sofa::helper::vector<unsigned int> _tArray)
        : core::topology::TopologyChange(core::topology::TETRAHEDRAREMOVED),
          removedTetrahedraArray(_tArray)
    { }

    virtual ~TetrahedraRemoved();

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedTetrahedraArray;
    }

    std::size_t getNbRemovedTetrahedra() const
    {
        return removedTetrahedraArray.size();
    }

public:
    sofa::helper::vector<unsigned int> removedTetrahedraArray;
};


/** indicates that some tetrahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * TetrahedraMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API TetrahedraMoved_Removing : public core::topology::TopologyChange
{
public:
    TetrahedraMoved_Removing (const sofa::helper::vector< unsigned int >& tetrahedronShell)
        : core::topology::TopologyChange (core::topology::TETRAHEDRAMOVED_REMOVING),
          tetrahedraAroundVertexMoved (tetrahedronShell)
    {}

    virtual ~TetrahedraMoved_Removing();
    
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return tetrahedraAroundVertexMoved; }

public:
    sofa::helper::vector< unsigned int > tetrahedraAroundVertexMoved;
};


/** indicates that some tetrahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * TetrahedraMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API TetrahedraMoved_Adding : public core::topology::TopologyChange
{
public:
    TetrahedraMoved_Adding (const sofa::helper::vector< unsigned int >& tetrahedronShell,
            const sofa::helper::vector< Topology::Tetrahedron >& tetrahedronArray)
        : core::topology::TopologyChange (core::topology::TETRAHEDRAMOVED_ADDING),
          tetrahedraAroundVertexMoved (tetrahedronShell), tetrahedronArray2Moved (tetrahedronArray)
    {}

    virtual ~TetrahedraMoved_Adding();
    
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return tetrahedraAroundVertexMoved; }
    const sofa::helper::vector< Topology::Tetrahedron >& getElementArray() const { return tetrahedronArray2Moved; }

public:
    sofa::helper::vector< unsigned int > tetrahedraAroundVertexMoved;
    const sofa::helper::vector< Topology::Tetrahedron > tetrahedronArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API TetrahedraRenumbering : public core::topology::TopologyChange
{
public:

    TetrahedraRenumbering()
        : core::topology::TopologyChange(core::topology::TETRAHEDRARENUMBERING)
    { }

    TetrahedraRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::topology::TopologyChange(core::topology::TETRAHEDRARENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    virtual ~TetrahedraRenumbering();

    const sofa::helper::vector<unsigned int> &getIndexArray() const { return indexArray; }

    const sofa::helper::vector<unsigned int> &getinv_IndexArray() const { return inv_indexArray; }

public:
    sofa::helper::vector<unsigned int> indexArray;
    sofa::helper::vector<unsigned int> inv_indexArray;
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Hexahedron Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two Hexahedra are being swapped */
class SOFA_CORE_API HexahedraIndicesSwap : public core::topology::TopologyChange
{
public:
    HexahedraIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::HEXAHEDRAINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

    virtual ~HexahedraIndicesSwap();

public:
    unsigned int index[2];
};


/** indicates that some hexahedra were added */
class SOFA_CORE_API HexahedraAdded : public core::topology::TopologyChange
{
public:
    HexahedraAdded(const unsigned int nT) : core::topology::TopologyChange(core::topology::HEXAHEDRAADDED),
        nHexahedra(nT)
    { }

    HexahedraAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Hexahedron >& _hexahedronArray,
            const sofa::helper::vector< unsigned int >& hexahedraIndex)
        : core::topology::TopologyChange(core::topology::HEXAHEDRAADDED),
          nHexahedra(nT),
          hexahedronArray(_hexahedronArray),
          hexahedronIndexArray(hexahedraIndex)
    { }

    HexahedraAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Hexahedron >& _hexahedronArray,
            const sofa::helper::vector< unsigned int >& hexahedraIndex,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::HEXAHEDRAADDED),
          nHexahedra(nT),
          hexahedronArray(_hexahedronArray),
          hexahedronIndexArray(hexahedraIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }
    
    HexahedraAdded(const unsigned int nT,
            const sofa::helper::vector< Topology::Hexahedron >& _hexahedronArray,
            const sofa::helper::vector< unsigned int >& hexahedraIndex,
            const sofa::helper::vector< HexahedronAncestorElem >& srcElems,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::HEXAHEDRAADDED),
          nHexahedra(nT),
          hexahedronArray(_hexahedronArray),
          hexahedronIndexArray(hexahedraIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs),
          ancestorElems(srcElems)
    { }

    virtual ~HexahedraAdded();

    unsigned int getNbAddedHexahedra() const
    {
        return nHexahedra;
    }

    unsigned int getNbAddedElements() const { return nHexahedra; }
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return hexahedronIndexArray; }
    const sofa::helper::vector< Topology::Hexahedron >& getElementArray() const { return hexahedronArray; }

public:
    unsigned int nHexahedra;
    sofa::helper::vector< Topology::Hexahedron > hexahedronArray;
    sofa::helper::vector< unsigned int > hexahedronIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
    sofa::helper::vector< HexahedronAncestorElem > ancestorElems;
};

/** indicates that some hexahedra are about to be removed */
class SOFA_CORE_API HexahedraRemoved : public core::topology::TopologyChange
{
public:
    HexahedraRemoved(const sofa::helper::vector<unsigned int> _tArray)
        : core::topology::TopologyChange(core::topology::HEXAHEDRAREMOVED),
          removedHexahedraArray(_tArray)
    { }

    virtual ~HexahedraRemoved();

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedHexahedraArray;
    }

    std::size_t getNbRemovedHexahedra() const
    {
        return removedHexahedraArray.size();
    }

public:
    sofa::helper::vector<unsigned int> removedHexahedraArray;
};


/** indicates that some hexahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * HexahedraMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class SOFA_CORE_API HexahedraMoved_Removing : public core::topology::TopologyChange
{
public:
    HexahedraMoved_Removing (const sofa::helper::vector< unsigned int >& hexahedronShell)
        : core::topology::TopologyChange (core::topology::HEXAHEDRAMOVED_REMOVING),
          hexahedraAroundVertexMoved (hexahedronShell)
    {}

    virtual ~HexahedraMoved_Removing();

    const sofa::helper::vector< unsigned int >& getIndexArray() const { return hexahedraAroundVertexMoved; }

public:
    sofa::helper::vector< unsigned int > hexahedraAroundVertexMoved;
};


/** indicates that some hexahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * HexahedraMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class SOFA_CORE_API HexahedraMoved_Adding : public core::topology::TopologyChange
{
public:
    HexahedraMoved_Adding (const sofa::helper::vector< unsigned int >& hexahedronShell,
            const sofa::helper::vector< Topology::Hexahedron >& hexahedronArray)
        : core::topology::TopologyChange (core::topology::HEXAHEDRAMOVED_ADDING),
          hexahedraAroundVertexMoved (hexahedronShell), hexahedronArray2Moved (hexahedronArray)
    {}

    virtual ~HexahedraMoved_Adding();
    
    const sofa::helper::vector< unsigned int >& getIndexArray() const { return hexahedraAroundVertexMoved; }
    const sofa::helper::vector< Topology::Hexahedron >& getElementArray() const { return hexahedronArray2Moved; }

public:
    sofa::helper::vector< unsigned int > hexahedraAroundVertexMoved;
    const sofa::helper::vector< Topology::Hexahedron > hexahedronArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class SOFA_CORE_API HexahedraRenumbering : public core::topology::TopologyChange
{
public:

    HexahedraRenumbering() : core::topology::TopologyChange(core::topology::HEXAHEDRARENUMBERING)
    { }

    HexahedraRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::topology::TopologyChange(core::topology::HEXAHEDRARENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    virtual ~HexahedraRenumbering();

    const sofa::helper::vector<unsigned int> &getIndexArray() const { return indexArray; }

    const sofa::helper::vector<unsigned int> &getinv_IndexArray() const { return inv_indexArray; }

public:
    sofa::helper::vector<unsigned int> indexArray;
    sofa::helper::vector<unsigned int> inv_indexArray;
};


} // namespace topology

} // namespace core

} // namespace sofa


#endif // SOFA_CORE_TOPOLOGY_TOPOLOGYCHANGE_H
