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
class Map2 : public Map1
{
protected:
	AttributeMultiVector<Dart>* m_phi2 ;

	void init() ;

public:
	typedef Map1 ParentMap;

	inline static unsigned int ORBIT_IN_PARENT(unsigned int o) { return o+5; }

	static const unsigned int IN_PARENT = 5 ;
	static const unsigned int DIMENSION = 2 ;

	static const unsigned int VERTEX_OF_PARENT = VERTEX+5;
	static const unsigned int EDGE_OF_PARENT = EDGE+5;

	Map2();

	virtual std::string mapTypeName() const;

	virtual unsigned int dimension() const;

	virtual void clear(bool removeAttrib);

	virtual void update_topo_shortcuts();

	virtual void compactTopoRelations(const std::vector<unsigned int>& oldnew);

	/*! @name Basic Topological Operators
	 * Access and Modification
	 *************************************************************************/

	virtual Dart newDart();

	Dart phi2(Dart d);

	template <int N>
	Dart phi(Dart d);

	Dart alpha0(Dart d);

	Dart alpha1(Dart d);

	Dart alpha_1(Dart d);

	/**
	 * prefer phi2_1 to alpha1 in algo if your want it to work in Map2 of Map3
	 */
	Dart phi2_1(Dart d);

	/**
	 * prefer phi21 to alpha_1 in algo if your want it to work in Map2 of Map3
	 */
	Dart phi12(Dart d);

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

	void rdfi(Dart t, DartMarker& m1, DartMarker& m2);

	/*! @name Generator and Deletor
	 *  To generate or delete faces in a 2-map
	 *************************************************************************/

	//@{
	//! Create an new polyline of nbEdges, i.e 2*nbEdges darts pairwise sewn by phi2
	/*! @param nbEdges the number of edges
	 *  @return return a dart of the face
	 */
	virtual Dart newPolyLine(unsigned int nbEdges) ;

	//! Create an new face of nbEdges
	/*! @param nbEdges the number of edges
	 *  @param withBoundary create the face and its boundary (default true)
	 *  @return return a dart of the face
	 */
	virtual Dart newFace(unsigned int nbEdges, bool withBoundary = true) ;

	//! Delete the face of d
	/*! @param d a dart of the face
	 *  @param withBoundary create or extend boundary face instead of fixed points (default true)
	 */
	virtual void deleteFace(Dart d, bool withBoundary = true) ;

	//! Delete a connected component of the map
	/*! @param d a dart of the connected component
	 */
	virtual void deleteCC(Dart d) ;

	//! Fill a hole with a face
	/*! \pre Dart d is boundary marked
	 *  @param d a dart of the face to fill
	 */
	virtual void fillHole(Dart d) ;

	//! Open the mesh Transforming a face in a hole
	/*! \pre Dart d is NOT boundary marked
	 *  @param d a dart of the face filled
	 */
	virtual void createHole(Dart d) ;
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
	virtual void splitVertex(Dart d, Dart e);

	//! Delete the vertex of d (works only for internal vertices)
	/*! Does not create a hole -> all the faces
	 * 	around the vertex are merged into one face
	 *  @param d a dart of the vertex to delete
	 *  @return a dart of the resulting face (NIL if the deletion has not been executed)
	 */
	virtual Dart deleteVertex(Dart d);

	//! Cut the edge of d by inserting a new vertex
	/*! @param d a dart of the edge to cut
	 *  @return a dart of the new vertex
	 */
	virtual Dart cutEdge(Dart d);

	//! Undo the cut of the edge of d
	/*! @param d a dart of the edge to uncut
	 *  @return true if the uncut has been executed, false otherwise
	 */
	virtual bool uncutEdge(Dart d);

	//! Collapse an edge (that is deleted) possibly merging its vertices
	/*! If delDegenerateFaces is true, the method checks that no degenerate
	 *  faces are built (faces with less than 3 edges). If it occurs the faces
	 *  are deleted and the adjacencies are updated (see collapseDegeneratedFace).
	 *  \warning This may produce two distinct vertices if the edge
	 *  was the only link between two border faces
	 *  @param d a dart in the deleted edge
	 *  @param delDegenerateFaces a boolean (default to true)
	 *  @return a dart of the resulting vertex
	 */
	virtual Dart collapseEdge(Dart d, bool delDegenerateFaces = true);

	/**
	 * Flip the edge of d (rotation in phi1 order)
	 * WARNING : Works only for non-border edges
	 * @param d a dart of the edge to flip
	 * @return true if the flip has been executed, false otherwise
	 */
	virtual bool flipEdge(Dart d);

	/**
	 * Flip the edge of d (rotation in phi_1 order)
	 * WARNING : Works only for non-border edges
	 * @param d a dart of the edge to flip
	 * @return true if the flipBack has been executed, false otherwise
	 */
	virtual bool flipBackEdge(Dart d);

	//!
	/*!
	 *
	 */
	void swapEdges(Dart d, Dart e);

	 //	 *  @param d dart of the vertex
	 //	 *  @param e dart of the edge
	 //	 */
	virtual void insertEdgeInVertex(Dart d, Dart e);
	 //
	 //	//! Remove an edge from a vertex orbit
	 //	/*! \pre Dart d must be phi2 sewed
	 //	 *  @param d the dart of the edge to remove from the vertex
	 //	 * @return true if the removal has been executed, false otherwise
	 //	 */
	virtual bool removeEdgeFromVertex(Dart d);

	//! Sew two oriented faces along oriented edges
	/*! \pre Edges of darts d & e MUST be boundary edges
	 *  @param d a dart of the first face
	 *  @param e a dart of the second face
	 *  @param withBoundary: if false, faces must have phi2 fixed points (only for construction: import/primitives)
	 */
	virtual void sewFaces(Dart d, Dart e, bool withBoundary = true);

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
	virtual void splitFace(Dart d, Dart e);

	//! Merge the two faces incident to the edge of d.
	/*! Works only for non-boundary edges.
	 *  \warning Darts of the edge of d no longer exist after the call
	 *  @param d a dart in the first face
	 *  @return true if the merge has been executed, false otherwise
	 */
	virtual bool mergeFaces(Dart d);

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
	virtual bool mergeVolumes(Dart d, Dart e, bool deleteFace = true);

	//! Split a surface into two disconnected surfaces along a edge path
	/*! @param vd a vector of darts
	 *  @param firstSideOpen : if false, one of the 2 sides of the surface remains closed (no hole)
	 *  @param secondSideOpen : if false, the other side of the surface remains closed (no hole)
	 */
	virtual void splitSurface(std::vector<Dart>& vd, bool firstSideClosed = true, bool secondSideClosed = true);
	//@}

	/*! @name Topological Queries
	 *  Return or set various topological information
	 *************************************************************************/

	//@{
	//! Test if dart d and e belong to the same oriented vertex
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameOrientedVertex(Dart d, Dart e) ;

	//! Test if dart d and e belong to the same vertex
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameVertex(Dart d, Dart e) ;

	//! Compute the number of edges of the vertex of d
	/*! @param d a dart
	 */
	unsigned int vertexDegree(Dart d) ;

	//! Check number of edges of the vertex of d with given parameter
	/*! @param d a dart
	 *	@param vd degree to compare with
	 *  @return  negative/null/positive if vertex degree is less/equal/greater than given degree
	 */
	int checkVertexDegree(Dart d, unsigned int vd);

	//! Tell if the vertex of d is on the boundary of the map
	/*! @param d a dart
	 */
	bool isBoundaryVertex(Dart d) ;

	/**
	 * find the dart of vertex that belong to the boundary
	 * return NIL if the vertex is not on the boundary
	 */
	Dart findBoundaryEdgeOfVertex(Dart d);

	/**
	 * find the dart of edge that belong to the boundary
	 * return NIL if the face is not on the boundary
	 */
	Dart findBoundaryEdgeOfFace(Dart d);

	//! Test if dart d and e belong to the same edge
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameEdge(Dart d, Dart e) ;

	/**
	 * tell if the edge of d is on the boundary of the map
	 */
	bool isBoundaryEdge(Dart d) ;

	//! Test if dart d and e belong to the same oriented face
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameOrientedFace(Dart d, Dart e);

	//! Test if dart d and e belong to the same face
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameFace(Dart d, Dart e) ;

	/**
	 * compute the number of edges of the face of d
	 */
	unsigned int faceDegree(Dart d) ;

	//! Check number of edges of the face of d with given parameter
	/*! @param d a dart
	 *	@param vd degree to compare with
	 *  @return  negative/null/positive if vertex degree is less/equal/greater than given degree
	 */
	int checkFaceDegree(Dart d, unsigned int le);

	/**
	 * tell if the face of d is on the boundary of the map
	 */
	bool isBoundaryFace(Dart d) ;

	//! Test if dart d and e belong to the same oriented volume
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameOrientedVolume(Dart d, Dart e) ;

	//! Test if dart d and e belong to the same volume
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameVolume(Dart d, Dart e) ;

	//! Compute the number of faces in the volume of d
	/*! @param d a dart
	 */
	unsigned int volumeDegree(Dart d);

	//! Check number of faces of the volume of d with given parameter
	/*! @param d a dart
	 *	@param vd degree to compare with
	 *  @return  negative/null/positive if volume degree is less/equal/greater than given degree
	 */
	int checkVolumeDegree(Dart d, unsigned int volDeg);

	// TODO a mettre en algo
	/**
	 * check if the mesh is triangular or not
	 * @return a boolean indicating if the mesh is triangular
	 */
	bool isTriangular() ;

	// TODO a mettre en algo
	/**
	 * Check if map is complete
	 * Should be executed after import
	 */
	virtual bool check();

	/**
	 * Check if a serie of darts is an oriented simple close path
	 */
	virtual bool checkSimpleOrientedPath(std::vector<Dart>& vd);
	//@}

	/*! @name Cell Functors
	 *  Apply functors to all darts of a cell
	 *************************************************************************/

	//@{
	//! Apply a functor on every dart of a vertex
	/*! @param d a dart of the vertex
	 *  @param f the functor to apply
	 */
	bool foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int thread = 0);

	//! Apply a functor on every dart of an edge
	/*! @param d a dart of the edge
	 *  @param f the functor to apply
	 */
	bool foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int thread = 0);

	//! Apply a functor on every dart of an face
	/*! @param d a dart of the volume
	 *  @param f the functor to apply
	 */
	bool foreach_dart_of_face(Dart d, FunctorType& f, unsigned int thread = 0);

	//! Apply a functor on every dart of an face
	/*! @param d a dart of the volume
	 *  @param f the functor to apply
	 */
	bool foreach_dart_of_volume(Dart d, FunctorType& f, unsigned int thread = 0);

	//! Apply a functor on every dart of a connected component
	/*! @param d a dart of the connected component
	 *  @param f the functor to apply
	 */
	bool foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread = 0);

	//! Apply a functor on every dart of a vertex of map1 representing the face of d
	/*! @param d a dart of the vertex
	 *  @param f the functor to apply
	 */
	bool foreach_dart_of_vertex1(Dart d, FunctorType& f, unsigned int thread = 0);

	//! Apply a functor on every dart of an edge of map1 representing the face of d
	/*! @param d a dart of the edge
	 *  @param f the functor to apply
	 */
	bool foreach_dart_of_edge1(Dart d, FunctorType& f, unsigned int thread = 0);

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
