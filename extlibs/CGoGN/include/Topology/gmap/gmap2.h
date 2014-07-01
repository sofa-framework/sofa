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

#ifndef __GMAP2_H__
#define __GMAP2_H__

#include "Topology/gmap/gmap1.h"

namespace CGoGN
{

/**
* The class of 2-GMap
*/
template <typename MAP_IMPL>
class GMap2 : public GMap1<MAP_IMPL>
{
protected:
	// protected copy constructor to prevent the copy of map
	GMap2(const GMap2<MAP_IMPL>& m):GMap1<MAP_IMPL>(m) {}

	void init() ;

public:
	typedef MAP_IMPL IMPL;
	typedef GMap1<MAP_IMPL> ParentMap;

	inline static unsigned int ORBIT_IN_PARENT(unsigned int o) { return o+5; }

	static const unsigned int IN_PARENT = 5 ;

	static const unsigned int VERTEX_OF_PARENT = VERTEX+5;
	static const unsigned int EDGE_OF_PARENT = EDGE+5;

	static const unsigned int DIMENSION = 2 ;

	GMap2();

	virtual std::string mapTypeName() const;

	virtual unsigned int dimension() const;

	virtual void clear(bool removeAttrib);

	virtual unsigned int getNbInvolutions() const;
	virtual unsigned int getNbPermutations() const;

	/*! @name Basic Topological Operators
	 * Access and Modification
	 *************************************************************************/

	Dart beta2(Dart d) const;

	template <int N>
	Dart beta(const Dart d) const;

	Dart phi2(Dart d) const;

	template <int N>
	Dart phi(const Dart d) const;

	Dart alpha0(Dart d) const;

	Dart alpha1(Dart d) const;

	Dart alpha_1(Dart d) const;

protected:
	void beta2sew(Dart d, Dart e);

	void beta2unsew(Dart d);

public:
	/*! @name Constructors and Destructors
	 *  To generate or delete cells in a 2-G-map
	 *************************************************************************/

	//@{
	//! Create an new face of nbEdges
	/*! @param nbEdges the number of edges
	 *  @param withBoudary create the face and its boundary (default true)
	 *  @return return a dart of the face
	 */
	Dart newFace(unsigned int nbEdges, bool withBoundary = true) ;

	//! Delete a face erasing all its darts
	/*! @param d a dart of the face
	 */
	void deleteFace(Dart d);

	//! Delete a connected component of the map
	/*! @param d a dart of the connected component
	 */
	void deleteCC(Dart d) ;

	//! Fill a hole with a face
	/*! \pre Dart d is boundary marked
	 *  @param d a dart of the face to fill
	 */
	void fillHole(Dart d) ;
	//@}

	/*! @name Topological Operators
	 *  Topological operations on 2-G-maps
	 *************************************************************************/

	//@{
	//! Split a vertex v between d and e inserting an edge after d and e
	/*! \pre Darts d & e MUST belong to the same oriented vertex
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
	Dart deleteVertex(Dart d) ;

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

	//! Collapse an edge (that is deleted) possibly merging its vertices
	/*! If delDegenerateFaces is true, the method checks that no degenerate
	 *  faces are build (faces with less than 3 edges). If it occurs the faces
	 *  are deleted and the adjacencies are updated (see deleteIfDegenerated).
	 *  \warning This may produce two distinct vertices if the edge
	 *  was the only link between two border faces
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

//	//! Insert an edge after a dart in the vertex orbit
//	/*! \pre Dart d and e MUST be different and belong to distinct face
//	 *  \pre Dart e must be phi2-linked with its phi_1 dart
//	 *  @param d dart of the vertex
//	 *  @param e dart of the edge
//	 */
//	virtual void insertEdgeInVertex(Dart d, Dart e);
//
//	//! Remove an edge from a vertex orbit
//	/*! \pre Dart d must be phi2 sewn
//	 *  @param d the dart of the edge to remove from the vertex
//	 */
//	virtual void removeEdgeFromVertex(Dart d);

	//! Sew two faces along an edge (pay attention to the orientation !)
	/*! \pre Edges of darts d & e MUST be boundary edges
	 *  @param d a dart of the first face
	 *  @param e a dart of the second face
	 *  @param withBoundary: if false, faces must have beta2 fixed points (only for construction: import/primitives)
	 */
	void sewFaces(Dart d, Dart e, bool withBoundary = true);

	//! Unsew two faces
	/*! \pre Edge of dart d MUST NOT be a boundary edge
	 *  @param d a dart of a face
	 */
	virtual void unsewFaces(Dart d);

	//! Delete a face if and only if it has one or two edges
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
	 * Insert a pair of sewed triangles in a vertex by exploding the edges of d1 and d2
	 * d1 and d2 belong to the same vertex
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
	bool mergeVolumes(Dart d, Dart e);

	//!
	/*!
	 *
	 */
	void splitSurface(std::vector<Dart>& vd, bool firstSideClosed = true, bool secondSideClosed = true);

	//@}

	/*! @name Topological Queries
	 *  Return or set various topological information
	 *************************************************************************/

	//@{
	//! Test if dart d and e belong to the same oriented vertex
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameOrientedVertex(Dart d, Dart e) const;

	//! Test if dart d and e belong to the same vertex
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameVertex(Dart d, Dart e) const;

	/**
	 * compute the number of edges of the vertex of d
	 */
	unsigned int vertexDegree(Dart d) const;

	//! Check number of edges of the vertex of d with given parameter
	/*! @param d a dart
	 *	@param vd degree to compare with
	 *  @return  negative/null/positive if vertex degree is less/equal/greater than given degree
	 */
	int checkVertexDegree(Dart d, unsigned int vd) const;

	//! tell if the vertex of d is on the boundary of the map
	/*! @param d a dart
	 */
	bool isBoundaryVertex(Dart d) const;

	/**
	 * find the dart of vertex that belong to the boundary
	 * return NIL if the vertex is not on the boundary
	 */
	Dart findBoundaryEdgeOfVertex(Dart d) const;

	//! Test if dart d and e belong to the same edge
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameEdge(Dart d, Dart e) const;

	/**
	 * tell if the edge of d is on the boundary of the map
	 */
	bool isBoundaryEdge(Dart d) const;

	//!Test if dart d and e belong to the same oriented face
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameOrientedFace(Dart d, Dart e) const;

	//! Test if dart d and e belong to the same face
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameFace(Dart d, Dart e) const;

	/**
	 * compute the number of edges of the face of d
	 */
	unsigned int faceDegree(Dart d) const;

	//! Check number of edges of the face of d with given parameter
	/*! @param d a dart
	 *	@param vd degree to compare with
	 *  @return  negative/null/positive if vertex degree is less/equal/greater than given degree
	 */
	int checkFaceDegree(Dart d, unsigned int le) const;

	/**
	 * tell if the face of d is adjacent to the boundary of the map
	 */
	bool isBoundaryAdjacentFace(Dart d) const;

	//! Test if dart d and e belong to the same oriented volume
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameOrientedVolume(Dart d, Dart e) const;

	//! Test if dart d and e belong to the same volume
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameVolume(Dart d, Dart e) const;

	//! Compute the number of faces in the volume of d
	/*! @param d a dart
	 */
	unsigned int volumeDegree(Dart d) const;

	//! Check number of faces of the volume of d with given parameter
	/*! @param d a dart
	 *	@param vd degree to compare with
	 *  @return  negative/null/positive if volume degree is less/equal/greater than given degree
	 */
	int checkVolumeDegree(Dart d, unsigned int volDeg) const;

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
	 * Check if a serie of darts is an oriented simple close path
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
	void foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f, unsigned int thread = 0) const ;
//	template <unsigned int ORBIT, typename FUNC>
//	void foreach_dart_of_orbit(Cell<ORBIT> c, FUNC& f, unsigned int thread = 0) const ;

	/**
	* Apply a functor on each dart of an oriented vertex
	* @param d a dart of the oriented vertex
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_oriented_vertex(Dart d, FUNC& f, unsigned int thread = 0) const;

	/**
	* Apply a functor on each dart of a vertex
	* @param d a dart of the vertex
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_vertex(Dart d, FUNC& f, unsigned int thread = 0) const;

	/**
	* Apply a functor on each dart of an oriented edge
	* @param d a dart of the oriented edge
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_oriented_edge(Dart d, FUNC& f, unsigned int thread = 0) const;

	/**
	* Apply a functor on each dart of an edge
	* @param d a dart of the oriented edge
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_edge(Dart d, FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on every dart of an oriented face
	/*! @param d a dart of the oriented face
	 *  @param f the functor to apply
	 */
	template <typename FUNC>
	void foreach_dart_of_oriented_face(Dart d, FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on every dart of a face
	/*! @param d a dart of the face
	 *  @param f the functor to apply
	 */
	template <typename FUNC>
	void foreach_dart_of_face(Dart d, FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on every dart of a volume
	/*! @param d a dart of the volume
	 *  @param f the functor to apply
	 */
	template <typename FUNC>
	void foreach_dart_of_volume(Dart d, FUNC& f, unsigned int thread = 0) const;

	/**
	* Apply a functor on each dart of a vertex
	* @param d a dart of the vertex
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_vertex1(Dart d, FUNC& f, unsigned int thread = 0) const;

	/**
	* Apply a functor on each dart of an edge
	* @param d a dart of the oriented edge
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_edge1(Dart d, FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on every dart of an oriented convex component
	/*! @param d a dart of the oriented convex component
	 *  @param f the functor to apply
	 */
	template <typename FUNC>
	void foreach_dart_of_oriented_cc(Dart d, FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on every dart of a convex component
	/*! @param d a dart of the convex component
	 *  @param f the functor to apply
	 */
	template <typename FUNC>
	void foreach_dart_of_cc(Dart d, FUNC& f, unsigned int thread = 0) const;

	//@}

	/*! @name Close map after import or creation
	 *  These functions must be used with care, generally only by import algorithms
	 *************************************************************************/

	//@{
	/**
	 * create a face of map1 marked as boundary
	 */
	Dart newBoundaryCycle(unsigned int nbE);

	//! Close a topological hole (a sequence of connected fixed point of beta2). DO NOT USE, only for import/creation algorithm
	/*! \pre dart d MUST be fixed point of beta2 relation
	 *  Add a face to the map that closes the hole.
	 *  @param d a dart of the hole (with beta2(d)==d)
	 *  @param forboundary tag the created face as boundary (default is true)
	 *  @return the degree of the created face
	 */
	virtual unsigned int closeHole(Dart d, bool forboundary = true);

	//! Close the map removing topological holes: DO NOT USE, only for import/creation algorithm
	/*! Add faces to the map that close every existing hole.
	 *  These faces are marked as boundary.
	 *  @return the number of closed holes
	 */
	unsigned int closeMap();
	//@}

	/*! @name Compute dual
	 * These functions compute the dual mesh
	 *************************************************************************/

	//@{
	//! Dual mesh computation
	/*!
	 */
	void computeDual();
	//@}
};

} // namespace CGoGN

#include "Topology/gmap/gmap2.hpp"

#endif
