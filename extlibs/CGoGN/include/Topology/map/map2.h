/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef __MAP2_H__
#define __MAP2_H__

#include "Topology/map/map1.h"

namespace CGoGN
{

/*! \brief The class of dual 2-dimensional combinatorial maps:
 *  set of oriented faces pairwise sewed by an adjacency relation.
 *  A dual 2-map represents close or open oriented 2-manifolds (surfaces).
 *  - A dual 2-map is made of darts linked by the phi1 permutation
 * 	and/or the phi2 one-to-one relation.
 *  - In this class darts are interpreted as oriented edges.
 *  - The phi1 relation defines oriented faces (see Map1)
 *  and faces may have arbitrary size (degenerated faces are accepted).
 *  - The phi2 relation links oriented faces along oriented edges.
 *  A set of phi2-linked faces represents a surface
 *  - Edges that have no phi2-link are border edges. If there exists
 *  such edges the map is open.
 *  - When every edge is phi2-linked, the map is closed. In this case
 *  some optimizations are enabled that speed up the processing of vertices.
 */
template <typename MAP_IMPL>
class Map2 : public Map1<MAP_IMPL>
{
protected:
    // protected copy constructor to prevent the copy of map
    Map2(const Map2<MAP_IMPL>& m):Map1<MAP_IMPL>(m) {}

    void init() ;

public:
    typedef MAP_IMPL IMPL;
    typedef Map1<MAP_IMPL> ParentMap;

    inline static unsigned int ORBIT_IN_PARENT(unsigned int o) { return o+5; }

    static const unsigned int IN_PARENT = 5 ;

    static const unsigned int VERTEX_OF_PARENT = VERTEX+5;
    static const unsigned int EDGE_OF_PARENT = EDGE+5;

    static const unsigned int DIMENSION = 2 ;

    Map2();

    virtual std::string mapTypeName() const;

    virtual unsigned int dimension() const;

    virtual void clear(bool removeAttrib);

    virtual unsigned int getNbInvolutions() const;
    virtual unsigned int getNbPermutations() const;

    /*! @name Basic Topological Operators
     * Access and Modification
     *************************************************************************/

    Dart phi2(Dart d) const;

    template <int N>
    Dart phi(Dart d) const;

    Dart alpha0(Dart d) const;

    Dart alpha1(Dart d) const;

    Dart alpha_1(Dart d) const;

    /**
     * prefer phi2_1 to alpha1 in algo if your want it to work in Map2 of Map3
     */
    Dart phi2_1(Dart d) const;

    /**
     * prefer phi21 to alpha_1 in algo if your want it to work in Map2 of Map3
     */
    Dart phi12(Dart d) const;

protected:
    //! Link dart d with dart e by an involution
    /*  @param d,e the darts to link
     *	- Before:	d->d and e->e
     *	- After:	d->e and e->d
     */
    void phi2sew(Dart d, Dart e);

    //! Unlink the current dart by an involution
    /*  @param d the dart to unlink
     * - Before:	d->e and e->d
     * - After:		d->d and e->e
     */
    void phi2unsew(Dart d);

public:
    /*! @name Generator and Deletor
     *  To generate or delete faces in a 2-map
     *************************************************************************/

    //@{
    //! Create an new polyline of nbEdges, i.e 2*nbEdges darts pairwise sewn by phi2
    /*! @param nbEdges the number of edges
     *  @return return a dart of the face
     */
    Dart newPolyLine(unsigned int nbEdges) ;

    //! Create an new face of nbEdges
    /*! @param nbEdges the number of edges
     *  @param withBoundary create the face and its boundary (default true)
     *  @return return a dart of the face
     */
    Dart newFace(unsigned int nbEdges, bool withBoundary = true) ;

    //! Delete the face of d
    /*! @param d a dart of the face
     *  @param withBoundary create or extend boundary face instead of fixed points (default true)
     */
    void deleteFace(Dart d) ;

    //! Delete a connected component of the map
    /*! @param d a dart of the connected component
     */
    void deleteCC(Dart d) ;

    //! Fill a hole with a face
    /*! \pre Dart d is boundary marked
     *  @param d a dart of the face to fill
     */
    void fillHole(Dart d) ;

    //! Open the mesh Transforming a face in a hole
    /*! \pre Dart d is NOT boundary marked
     *  @param d a dart of the face filled
     */
    void createHole(Dart d) ;
    //@}

    /*! @name Topological Operators
     *  Topological operations on 2-maps
     *************************************************************************/

    //@{
    //! Split a vertex v between d and e inserting an edge after d and e
    /*! \pre Darts d & e MUST belong to the same vertex
     *  @param d first dart in vertex v
     *  @param e second dart in vertex v
     */
    void splitVertex(Dart d, Dart e);

    //! Delete the vertex of d (works only for internal vertices)
    /*! Does not create a hole -> all the faces
     * 	around the vertex are merged into one face
     *  @param d a dart of the vertex to delete
     *  @return a dart of the resulting face (NIL if the deletion has not been executed)
     */
    Dart deleteVertex(Dart d);

    //! Cut the edge of d by inserting a new vertex
    /*! @param d a dart of the edge to cut
     *  @return a dart of the new vertex
     */
    Dart cutEdge(Dart d);

    //! Undo the cut of the edge of d
    /*! @param d a dart of the edge to uncut
     *  @return true if the uncut has been executed, false otherwise
     */
    bool uncutEdge(Dart d);

    //! Collapse an edge (that is deleted) by merging its vertices
    /*! If delDegenerateFaces is true, the method checks that no degenerate
     *  faces are built (faces with less than 3 edges). If it occurs the faces
     *  are deleted and the adjacencies are updated (see collapseDegeneratedFace).
     *  @param d a dart in the deleted edge
     *  @param delDegenerateFaces a boolean (default to true)
     *  @return a dart of the resulting vertex
     */
    Dart collapseEdge(Dart d, bool delDegenerateFaces = true);

    /**
     * Flip the edge of d (rotation in phi1 order)
     * WARNING : Works only for non-border edges
     * @param d a dart of the edge to flip
     * @return true if the flip has been executed, false otherwise
     */
    bool flipEdge(Dart d);

    /**
     * Flip the edge of d (rotation in phi_1 order)
     * WARNING : Works only for non-border edges
     * @param d a dart of the edge to flip
     * @return true if the flipBack has been executed, false otherwise
     */
    bool flipBackEdge(Dart d);

    //!
    /*!
     *
     */
    void swapEdges(Dart d, Dart e);

    //	 *  @param d dart of the vertex
    //	 *  @param e dart of the edge
    //	 */
    void insertEdgeInVertex(Dart d, Dart e);
    //
    //	//! Remove an edge from a vertex orbit
    //	/*! \pre Dart d must be phi2 sewed
    //	 *  @param d the dart of the edge to remove from the vertex
    //	 * @return true if the removal has been executed, false otherwise
    //	 */
    bool removeEdgeFromVertex(Dart d);

    //! Sew two oriented faces along oriented edges
    /*! \pre Edges of darts d & e MUST be boundary edges
     *  @param d a dart of the first face
     *  @param e a dart of the second face
     *  @param withBoundary: if false, faces must have phi2 fixed points (only for construction: import/primitives)
     */
    void sewFaces(Dart d, Dart e, bool withBoundary = true);

    //! Unsew two oriented faces
    /*! \pre Edge of dart d MUST NOT be a boundary edge
     *  @param d a dart of a face
     */
    virtual void unsewFaces(Dart d, bool withBoundary = true);

    //! Delete an oriented face if and only if it has one or two edges
    /*! If the face is sewed to two distinct adjacent faces,
     *  then those two faces are sewed
     *  @param d a dart of the face
     *  @return true if the collapse has been executed, false otherwise
     */
    virtual bool collapseDegeneratedFace(Dart d);

    //! Split a face f between d and e inserting an edge between vertices d & e
    /*! \pre Darts d & e MUST belong to the same face
     *  @param d first dart in face f
     *  @param e second dart in face f
     */
    void splitFace(Dart d, Dart e);

    //! Merge the two faces incident to the edge of d.
    /*! Works only for non-boundary edges.
     *  \warning Darts of the edge of d no longer exist after the call
     *  @param d a dart in the first face
     *  @return true if the merge has been executed, false otherwise
     */
    bool mergeFaces(Dart d);

    /**
     * Extract a pair of sewed triangles and sew their adjacent faces
     * d is a dart of the common edge of the pair of triangles
     */
    void extractTrianglePair(Dart d) ;

    /**
     * Insert a pair of sewed triangles in a vertex by exploding the edges of v1 and v2
     * v1 and v2 belong to the same vertex
     * d is a dart of the common edge of the pair of triangles
     */
    void insertTrianglePair(Dart d, Dart v1, Dart v2) ;

    //! Merge two volumes along two faces.
    /*! Works only if the two faces have the same number of edges.
     *  The faces adjacent to the two given faces are pairwise sewed
     *  then the 2 faces are deleted.
     *  If the two faces belong to different surfaces, the surfaces are merged,
     *  else a handle is created that increases the genus of the surface.
     *  @param d a dart of the first face
     *  @param e a dart of the second face
     *  @return true if the merge has been executed, false otherwise
     */
    bool mergeVolumes(Dart d, Dart e, bool deleteFace = true);

    //! Split a surface into two disconnected surfaces along a edge path
    /*! @param vd a vector of darts
     *  @param firstSideOpen : if false, one of the 2 sides of the surface remains closed (no hole)
     *  @param secondSideOpen : if false, the other side of the surface remains closed (no hole)
     */
    void splitSurface(std::vector<Dart>& vd, bool firstSideClosed = true, bool secondSideClosed = true);
    //@}

    /*! @name Topological Queries
     *  Return or set various topological information
     *************************************************************************/

    //@{
    //! Test if vertices v1 and v2 represent the same oriented vertex
    /*! @param v1 a vertex
     *  @param v2 a vertex
     */
    bool sameOrientedVertex(Vertex v1, Vertex v2) const;

    //! Test if vertices v1 and v2 represent the same vertex
    /*! @param v1 a vertex
     *  @param v2 a vertex
     */
    bool sameVertex(Vertex v1, Vertex v2) const;

    //! Compute the number of edges of the vertex v
    /*! @param v a vertex
     */
    unsigned int vertexDegree(Vertex v) const;

    //! Check number of edges of the vertex v with given parameter
    /*! @param v a vertex
     *	@param vd degree to compare with
     *  @return negative/null/positive if vertex degree is less/equal/greater than given degree
     */
    int checkVertexDegree(Vertex v, unsigned int vd) const;

    //! Tell if the vertex v is on the boundary of the map
    /*! @param v a vertex
     */
    bool isBoundaryVertex(Vertex v) const;

    /**
     * find the dart of vertex v that belongs to the boundary
     * return NIL if the vertex is not on the boundary
     */
    Dart findBoundaryEdgeOfVertex(Vertex v) const;

    //! Test if edges e1 and e2 represent the same edge
    /*! @param e1 an edge
     *  @param e2 an edge
     */
    bool sameEdge(Edge e1, Edge e2) const;

    /**
     * tell if the edge e is on the boundary of the map
     */
    bool isBoundaryEdge(Edge e) const;

    //! Test if faces f1 and f2 represent the same oriented face
    /*! @param f1 a face
     *  @param f2 a face
     */
    bool sameOrientedFace(Face f1, Face f2) const;

    //! Test if faces f1 and f2 represent the same face
    /*! @param f1 a face
     *  @param f2 a face
     */
    bool sameFace(Face f1, Face f2) const;

    /**
     * compute the number of edges of the face f
     */
    unsigned int faceDegree(Face f) const;

    //! Check number of edges of the face f with given parameter
    /*! @param f a face
     *	@param fd degree to compare with
     *  @return negative/null/positive if face degree is less/equal/greater than given degree
     */
    int checkFaceDegree(Face f, unsigned int fd) const;

    /**
     * tell if the face f is incident to the boundary of the map
     */
    bool isFaceIncidentToBoundary(Face f) const;

    /**
     * find the dart of face f that belongs to the boundary
     * return NIL if the face is not incident to the boundary
     */
    Dart findBoundaryEdgeOfFace(Face f) const;

    //! Test if volumes v1 and v2 represent the same oriented volume
    /*! @param d a dart
     *  @param e a dart
     */
    bool sameOrientedVolume(Vol v1, Vol v2) const;

    //! Test if volumes v1 and v2 represent the same volume
    /*! @param d a dart
     *  @param e a dart
     */
    bool sameVolume(Vol v1, Vol v2) const;

    //! Compute the number of faces in the volume v
    /*! @param d a dart
     */
    unsigned int volumeDegree(Vol v) const;

    //! Check number of faces of the volume v with given parameter
    /*! @param v a volume
     *	@param vd degree to compare with
     *  @return negative/null/positive if volume degree is less/equal/greater than given degree
     */
    int checkVolumeDegree(Vol v, unsigned int vd) const;

    // TODO a mettre en algo
    /**
     * check if the mesh is triangular or not
     * @return a boolean indicating if the mesh is triangular
     */
    bool isTriangular() const;

    // TODO a mettre en algo
    /**
     * Check if map is complete
     * Should be executed after import
     */
    virtual bool check() const;

    /**
     * Check if a serie of edges is an oriented simple close path
     */
    virtual bool checkSimpleOrientedPath(std::vector<Dart>& vd);
    //@}

    /*! @name Cell Functors
     *  Apply functors to all darts of a cell
     *************************************************************************/

    //@{
    //! Apply a function on every dart of an orbit
    /*! @param c a cell
     *  @param f a function
     */
    template <unsigned int ORBIT, typename FUNC>
    void foreach_dart_of_orbit(Cell<ORBIT> c, const FUNC& f, unsigned int thread = 0) const ;
    //    template <unsigned int ORBIT, typename FUNC>
    //    void foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f, unsigned int thread = 0) const ;

    //! Apply a functor on every dart of a vertex
    /*! @param d a dart of the vertex
     *  @param f the functor to apply
     */
    template <typename FUNC>
    void foreach_dart_of_vertex(Dart d, const FUNC& f, unsigned int thread = 0) const;

    //! Apply a functor on every dart of an edge
    /*! @param d a dart of the edge
     *  @param f the functor to apply
     */
    template <typename FUNC>
    void foreach_dart_of_edge(Dart d, const FUNC& f, unsigned int thread = 0) const;

    //! Apply a functor on every dart of a face
    /*! @param d a dart of the volume
     *  @param f the functor to apply
     */
    template <typename FUNC>
    void foreach_dart_of_face(Dart d, const FUNC& f, unsigned int thread = 0) const;

    //! Apply a functor on every dart of a volume
    /*! @param d a dart of the volume
     *  @param f the functor to apply
     */
    template <typename FUNC>
    void foreach_dart_of_volume(Dart d, const FUNC& f, unsigned int thread = 0) const;

    //! Apply a functor on every dart of a vertex of map1 representing the face of d
    /*! @param d a dart of the vertex
     *  @param f the functor to apply
     */
    template <typename FUNC>
    void foreach_dart_of_vertex1(Dart d, const FUNC& f, unsigned int thread = 0) const;

    //! Apply a functor on every dart of an edge of map1 representing the face of d
    /*! @param d a dart of the edge
     *  @param f the functor to apply
     */
    template <typename FUNC>
    void foreach_dart_of_edge1(Dart d, const FUNC& f, unsigned int thread = 0) const;

    //! Apply a functor on every dart of a connected component
    /*! @param d a dart of the connected component
     *  @param f the functor to apply
     */
    template <typename FUNC>
    void foreach_dart_of_cc(Dart d, const FUNC& f, unsigned int thread = 0) const;

    //@}

    /*! @name Close map after import or creation
     *  These functions must be used with care, generally only by import algorithms
     *************************************************************************/

    //@{
    /**
     * create a face of map1 marked as boundary
     */
    Dart newBoundaryCycle(unsigned int nbE);

    //! Close a topological hole (a sequence of connected fixed point of phi2). DO NOT USE, only for import/creation algorithm
    /*! \pre dart d MUST be fixed point of phi2 relation
     *  Add a face to the map that closes the hole.
     *  @param d a dart of the hole (with phi2(d)==d)
     *  @param forboundary tag the created face as boundary (default is true)
     *  @return the degree of the created face
     */
    virtual unsigned int closeHole(Dart d, bool forboundary = true);

    //! Close the map removing topological holes: DO NOT USE, only for import/creation algorithm
    /*! Add faces to the map that close every existing hole.
     *  These faces are marked as boundary.
     *  @return the number of closed holes
     */
    unsigned int closeMap(bool forboundary = true);
    //@}

    /*! @name Compute dual
     * These functions compute the dual mesh
     *************************************************************************/

    //@{
    //! Reverse the orientation of the map
    /*!
     */
    void reverseOrientation();

    //! Dual mesh computation (open or closed)
    /*! Crop the infinite faces of open meshes
     */
    void computeDual();
    //@}
};

} // namespace CGoGN

#include "Topology/map/map2.hpp"

#endif
