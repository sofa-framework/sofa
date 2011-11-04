/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGYCHANGE_H
#define SOFA_CORE_TOPOLOGY_TOPOLOGYCHANGE_H

#include <sofa/core/topology/Topology.h>

namespace sofa
{

namespace core
{

namespace topology
{

using namespace sofa::helper;


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



/** \brief Base class to indicate a topology change occurred.
*
* All topological changes taking place in a given BaseTopology will issue a TopologyChange in the
* BaseTopology's changeList, so that BasicTopologies mapped to it can know what happened and decide how to
* react.
* Classes inheriting from this one describe a given topolopy change (e.g. RemovedPoint, AddedEdge, etc).
* The exact type of topology change is given by member changeType.
*/
class TopologyChange
{
public:
    /** \ brief Destructor.
        *
        * Must be virtual for TopologyChange to be a Polymorphic type.
        */
    virtual ~TopologyChange() {}

    /** \brief Returns the code of this TopologyChange. */
    TopologyChangeType getChangeType() const { return m_changeType;}


    /// Output empty stream
    inline friend std::ostream& operator<< ( std::ostream& os, const TopologyChange* /*t*/ )
    {
        return os;
    }

    /// Input empty stream
    inline friend std::istream& operator>> ( std::istream& in, const TopologyChange* /*t*/ )
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
class EndingEvent : public core::topology::TopologyChange
{
public:
    EndingEvent()
        : core::topology::TopologyChange(core::topology::ENDING_EVENT)
    {}
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   Point Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two points are being swapped */
class PointsIndicesSwap : public core::topology::TopologyChange
{
public:
    PointsIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::POINTSINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

public:
    unsigned int index[2];
};


/** indicates that some points were added */
class PointsAdded : public core::topology::TopologyChange
{
public:

    PointsAdded(const unsigned int nV) : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV)
    { }

    PointsAdded(const unsigned int nV,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double       > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::POINTSADDED)
        , nVertices(nV), ancestorsList(ancestors), coefs(baryCoefs)
    { }

    unsigned int getNbAddedVertices() const {return nVertices;}

public:
    unsigned int nVertices;
    sofa::helper::vector< unsigned int > pointIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double       > > coefs;
};


/** indicates that some points are about to be removed */
class PointsRemoved : public core::topology::TopologyChange
{
public:
    PointsRemoved(const sofa::helper::vector<unsigned int>& _vArray) : core::topology::TopologyChange(core::topology::POINTSREMOVED),
        removedVertexArray(_vArray)
    { }

    const sofa::helper::vector<unsigned int> &getArray() const { return removedVertexArray;	}

public:
    sofa::helper::vector<unsigned int> removedVertexArray;
};


/** indicates that the indices of all points have been renumbered */
class PointsRenumbering : public core::topology::TopologyChange
{
public:

    PointsRenumbering() : core::topology::TopologyChange(core::topology::POINTSRENUMBERING)
    { }

    PointsRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::topology::TopologyChange(core::topology::POINTSRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    const sofa::helper::vector<unsigned int> &getIndexArray() const { return indexArray; }

    const sofa::helper::vector<unsigned int> &getinv_IndexArray() const { return inv_indexArray; }

public:
    sofa::helper::vector<unsigned int> indexArray;
    sofa::helper::vector<unsigned int> inv_indexArray;
};


/** indicates that some points were moved */
class PointsMoved : public core::topology::TopologyChange
{
public:

    PointsMoved(const sofa::helper::vector<unsigned int>& indices,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::topology::TopologyChange(core::topology::POINTSMOVED)
        , indicesList(indices), ancestorsList(ancestors), baryCoefsList(baryCoefs)
    {}

public:
    sofa::helper::vector<unsigned int> indicesList;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > baryCoefsList;
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////   Edge Event Implementation   /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** indicates that the indices of two edges are being swapped */
class EdgesIndicesSwap : public core::topology::TopologyChange
{
public:
    EdgesIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::EDGESINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

public:
    unsigned int index[2];
};


/** indicates that some edges were added */
class EdgesAdded : public core::topology::TopologyChange
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

    virtual ~EdgesAdded() {}

    unsigned int getNbAddedEdges() const { return nEdges;}
    /*	const sofa::helper::vector<unsigned int> &getArray() const
        {
                return edgeIndexArray;
        }*/
    const sofa::helper::vector< Topology::Edge > &getArray() const
    {
        return edgeArray;
    }

public:
    unsigned int nEdges;
    sofa::helper::vector< Topology::Edge > edgeArray;
    sofa::helper::vector< unsigned int > edgeIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
};


/** indicates that some edges are about to be removed */
class EdgesRemoved : public core::topology::TopologyChange
{
public:
    EdgesRemoved(const sofa::helper::vector<unsigned int> _eArray) : core::topology::TopologyChange(core::topology::EDGESREMOVED),
        removedEdgesArray(_eArray)
    {}

    ~EdgesRemoved() {}

    virtual const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedEdgesArray;
    }

    virtual unsigned int getNbRemovedEdges() const
    {
        return removedEdgesArray.size();
    }

public:
    sofa::helper::vector<unsigned int> removedEdgesArray;
};


/** indicates that some edges are about to be moved (i.e one or both of their vertices have just been moved)
 * EdgesMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class EdgesMoved_Removing : public core::topology::TopologyChange
{
public:
    EdgesMoved_Removing (const sofa::helper::vector< unsigned int >& edgeShell) : core::topology::TopologyChange (core::topology::EDGESMOVED_REMOVING),
        edgesAroundVertexMoved (edgeShell)
    {}

public:
    sofa::helper::vector< unsigned int > edgesAroundVertexMoved;
};


/** indicates that some edges are about to be moved (i.e one or both of their vertices have just been moved)
 * EdgesMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class EdgesMoved_Adding : public core::topology::TopologyChange
{
public:
    EdgesMoved_Adding (const sofa::helper::vector< unsigned int >& edgeShell,
            const sofa::helper::vector< Topology::Edge >& edgeArray)
        : core::topology::TopologyChange (core::topology::EDGESMOVED_ADDING),
          edgesAroundVertexMoved (edgeShell), edgeArray2Moved (edgeArray)
    {}

public:
    sofa::helper::vector< unsigned int > edgesAroundVertexMoved;
    sofa::helper::vector< Topology::Edge > edgeArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class EdgesRenumbering : public core::topology::TopologyChange
{
public:
    EdgesRenumbering() : core::topology::TopologyChange(core::topology::EDGESRENUMBERING)
    { }

    EdgesRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::topology::TopologyChange(core::topology::EDGESRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

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
class TrianglesIndicesSwap : public core::topology::TopologyChange
{
public:
    TrianglesIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::TRIANGLESINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

public:
    unsigned int index[2];
};


/** indicates that some triangles were added */
class TrianglesAdded : public core::topology::TopologyChange
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

public:
    unsigned int nTriangles;
    sofa::helper::vector< Topology::Triangle > triangleArray;
    sofa::helper::vector< unsigned int > triangleIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
};


/** indicates that some triangles are about to be removed */
class TrianglesRemoved : public core::topology::TopologyChange
{
public:
    TrianglesRemoved(const sofa::helper::vector<unsigned int> _tArray) : core::topology::TopologyChange(core::topology::TRIANGLESREMOVED),
        removedTrianglesArray(_tArray)
    {}

    unsigned int getNbRemovedTriangles() const
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
class TrianglesMoved_Removing : public core::topology::TopologyChange
{
public:
    TrianglesMoved_Removing (const sofa::helper::vector< unsigned int >& triangleShell)
        : core::topology::TopologyChange (core::topology::TRIANGLESMOVED_REMOVING),
          trianglesAroundVertexMoved (triangleShell)
    {}

public:
    sofa::helper::vector< unsigned int > trianglesAroundVertexMoved;
};


/** indicates that some triangles are about to be moved (i.e some/all of their vertices have just been moved)
 * TrianglesMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class TrianglesMoved_Adding : public core::topology::TopologyChange
{
public:
    TrianglesMoved_Adding (const sofa::helper::vector< unsigned int >& triangleShell,
            const sofa::helper::vector< Topology::Triangle >& triangleArray)
        : core::topology::TopologyChange (core::topology::TRIANGLESMOVED_ADDING),
          trianglesAroundVertexMoved (triangleShell), triangleArray2Moved (triangleArray)
    {}

public:
    sofa::helper::vector< unsigned int > trianglesAroundVertexMoved;
    const sofa::helper::vector< Topology::Triangle > triangleArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class TrianglesRenumbering : public core::topology::TopologyChange
{
public:

    TrianglesRenumbering() : core::topology::TopologyChange(core::topology::TRIANGLESRENUMBERING)
    { }

    TrianglesRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::topology::TopologyChange(core::topology::TRIANGLESRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

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
class QuadsIndicesSwap : public core::topology::TopologyChange
{
public:
    QuadsIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::QUADSINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

public:
    unsigned int index[2];
};


/** indicates that some quads were added */
class QuadsAdded : public core::topology::TopologyChange
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

public:
    unsigned int nQuads;
    sofa::helper::vector< Topology::Quad > quadArray;
    sofa::helper::vector< unsigned int > quadIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
};


/** indicates that some quads are about to be removed */
class QuadsRemoved : public core::topology::TopologyChange
{
public:
    QuadsRemoved(const sofa::helper::vector<unsigned int> _qArray) : core::topology::TopologyChange(core::topology::QUADSREMOVED),
        removedQuadsArray(_qArray)
    { }

    unsigned int getNbRemovedQuads() const
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
class QuadsMoved_Removing : public core::topology::TopologyChange
{
public:
    QuadsMoved_Removing (const sofa::helper::vector< unsigned int >& quadShell) : core::topology::TopologyChange (core::topology::QUADSMOVED_REMOVING),
        quadsAroundVertexMoved (quadShell)
    {}

public:
    sofa::helper::vector< unsigned int > quadsAroundVertexMoved;
};


/** indicates that some quads are about to be moved (i.e some/all of their vertices have just been moved)
 * QuadsMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class QuadsMoved_Adding : public core::topology::TopologyChange
{
public:
    QuadsMoved_Adding (const sofa::helper::vector< unsigned int >& quadShell,
            const sofa::helper::vector< Topology::Quad >& quadArray)
        : core::topology::TopologyChange (core::topology::QUADSMOVED_ADDING),
          quadsAroundVertexMoved (quadShell), quadArray2Moved (quadArray)
    {}

public:
    sofa::helper::vector< unsigned int > quadsAroundVertexMoved;
    const sofa::helper::vector< Topology::Quad > quadArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class QuadsRenumbering : public core::topology::TopologyChange
{
public:

    QuadsRenumbering() : core::topology::TopologyChange(core::topology::QUADSRENUMBERING)
    { }

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
class TetrahedraIndicesSwap : public core::topology::TopologyChange
{
public:
    TetrahedraIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::TETRAHEDRAINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

public:
    unsigned int index[2];
};


/** indicates that some tetrahedra were added */
class TetrahedraAdded : public core::topology::TopologyChange
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

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return tetrahedronIndexArray;
    }

    unsigned int getNbAddedTetrahedra() const
    {
        return nTetrahedra;
    }

public:
    unsigned int nTetrahedra;
    sofa::helper::vector< Topology::Tetrahedron > tetrahedronArray;
    sofa::helper::vector< unsigned int > tetrahedronIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
};

/** indicates that some tetrahedra are about to be removed */
class TetrahedraRemoved : public core::topology::TopologyChange
{
public:
    TetrahedraRemoved(const sofa::helper::vector<unsigned int> _tArray)
        : core::topology::TopologyChange(core::topology::TETRAHEDRAREMOVED),
          removedTetrahedraArray(_tArray)
    { }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedTetrahedraArray;
    }

    unsigned int getNbRemovedTetrahedra() const
    {
        return removedTetrahedraArray.size();
    }

public:
    sofa::helper::vector<unsigned int> removedTetrahedraArray;
};


/** indicates that some tetrahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * TetrahedraMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class TetrahedraMoved_Removing : public core::topology::TopologyChange
{
public:
    TetrahedraMoved_Removing (const sofa::helper::vector< unsigned int >& tetrahedronShell)
        : core::topology::TopologyChange (core::topology::TETRAHEDRAMOVED_REMOVING),
          tetrahedraAroundVertexMoved (tetrahedronShell)
    {}

public:
    sofa::helper::vector< unsigned int > tetrahedraAroundVertexMoved;
};


/** indicates that some tetrahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * TetrahedraMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class TetrahedraMoved_Adding : public core::topology::TopologyChange
{
public:
    TetrahedraMoved_Adding (const sofa::helper::vector< unsigned int >& tetrahedronShell,
            const sofa::helper::vector< Topology::Tetrahedron >& tetrahedronArray)
        : core::topology::TopologyChange (core::topology::TETRAHEDRAMOVED_ADDING),
          tetrahedraAroundVertexMoved (tetrahedronShell), tetrahedronArray2Moved (tetrahedronArray)
    {}

public:
    sofa::helper::vector< unsigned int > tetrahedraAroundVertexMoved;
    const sofa::helper::vector< Topology::Tetrahedron > tetrahedronArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class TetrahedraRenumbering : public core::topology::TopologyChange
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
class HexahedraIndicesSwap : public core::topology::TopologyChange
{
public:
    HexahedraIndicesSwap(const unsigned int i1,const unsigned int i2) : core::topology::TopologyChange(core::topology::HEXAHEDRAINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

public:
    unsigned int index[2];
};


/** indicates that some hexahedra were added */
class HexahedraAdded : public core::topology::TopologyChange
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

    unsigned int getNbAddedHexahedra() const
    {
        return nHexahedra;
    }

public:
    unsigned int nHexahedra;
    sofa::helper::vector< Topology::Hexahedron > hexahedronArray;
    sofa::helper::vector< unsigned int > hexahedronIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
};

/** indicates that some hexahedra are about to be removed */
class HexahedraRemoved : public core::topology::TopologyChange
{
public:
    HexahedraRemoved(const sofa::helper::vector<unsigned int> _tArray)
        : core::topology::TopologyChange(core::topology::HEXAHEDRAREMOVED),
          removedHexahedraArray(_tArray)
    { }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedHexahedraArray;
    }

    unsigned int getNbRemovedHexahedra() const
    {
        return removedHexahedraArray.size();
    }

public:
    sofa::helper::vector<unsigned int> removedHexahedraArray;
};


/** indicates that some hexahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * HexahedraMoved_Removing First part, remove element concerned to force object to recompute global state with current positions
 */
class HexahedraMoved_Removing : public core::topology::TopologyChange
{
public:
    HexahedraMoved_Removing (const sofa::helper::vector< unsigned int >& hexahedronShell)
        : core::topology::TopologyChange (core::topology::HEXAHEDRAMOVED_REMOVING),
          hexahedraAroundVertexMoved (hexahedronShell)
    {}

public:
    sofa::helper::vector< unsigned int > hexahedraAroundVertexMoved;
};


/** indicates that some hexahedra are about to be moved (i.e some/all of their vertices have just been moved)
 * HexahedraMoved_Adding Second part, recompute state of all elements previously removed, with new positions points
 */
class HexahedraMoved_Adding : public core::topology::TopologyChange
{
public:
    HexahedraMoved_Adding (const sofa::helper::vector< unsigned int >& hexahedronShell,
            const sofa::helper::vector< Topology::Hexahedron >& hexahedronArray)
        : core::topology::TopologyChange (core::topology::HEXAHEDRAMOVED_ADDING),
          hexahedraAroundVertexMoved (hexahedronShell), hexahedronArray2Moved (hexahedronArray)
    {}

public:
    sofa::helper::vector< unsigned int > hexahedraAroundVertexMoved;
    const sofa::helper::vector< Topology::Hexahedron > hexahedronArray2Moved;
};


/** indicates that the indices of all points have been renumbered */
class HexahedraRenumbering : public core::topology::TopologyChange
{
public:

    HexahedraRenumbering() : core::topology::TopologyChange(core::topology::HEXAHEDRARENUMBERING)
    { }

    HexahedraRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::topology::TopologyChange(core::topology::HEXAHEDRARENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

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
