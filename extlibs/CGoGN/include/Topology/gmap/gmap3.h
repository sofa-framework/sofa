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

#ifndef __GMAP3_H__
#define __GMAP3_H__

#include "Topology/gmap/gmap2.h"

namespace CGoGN
{

/**
* The class of 3-GMap
*/
template <typename MAP_IMPL>
class GMap3 : public GMap2<MAP_IMPL>
{
protected:
	// protected copy constructor to prevent the copy of map
	GMap3(const GMap3<MAP_IMPL>& m):GMap2<MAP_IMPL>(m) {}

	void init() ;

public:
	typedef MAP_IMPL IMPL;
	typedef GMap2<MAP_IMPL> ParentMap;

	inline static unsigned int ORBIT_IN_PARENT(unsigned int o) { return o+7; }
	inline static unsigned int ORBIT_IN_PARENT2(unsigned int o) { return o+5; }

	static const unsigned int IN_PARENT = 7 ;
	static const unsigned int IN_PARENT2 = 5 ;

	static const unsigned int VERTEX_OF_PARENT = VERTEX+7;
	static const unsigned int EDGE_OF_PARENT = EDGE+7;
	static const unsigned int FACE_OF_PARENT = FACE+7;

	static const unsigned int VERTEX_OF_PARENT2 = VERTEX+5;
	static const unsigned int EDGE_OF_PARENT2 = EDGE+5;

	static const unsigned int DIMENSION = 3 ;

	GMap3();

	virtual std::string mapTypeName() const;

	virtual unsigned int dimension() const;

	virtual void clear(bool removeAttrib);

	virtual unsigned int getNbInvolutions() const;
	virtual unsigned int getNbPermutations() const;

	/*! @name Basic Topological Operators
	 * Access and Modification
	 *************************************************************************/

	Dart beta3(Dart d) const;

	template <int N>
	Dart beta(const Dart d) const;

	Dart phi3(Dart d) const;

	template <int N>
	Dart phi(const Dart d) const;

	Dart alpha0(Dart d) const;

	Dart alpha1(Dart d) const;

	Dart alpha2(Dart d) const;

	Dart alpha_2(Dart d) const;

protected:
	void beta3sew(Dart d, Dart e);

	void beta3unsew(Dart d);

public:
	/*! @name Generator and Deletor
	 *  To generate or delete volumes in a 3-G-map
	 *************************************************************************/

	//@{
	//! Delete a volume erasing all its darts.
	/*! The phi3-links around the volume are removed
	 *  @param d a dart of the volume
	 */
	void deleteVolume(Dart d);

	//! Fill a hole with a volume
	/*! \pre Dart d is boundary marked
	 *  @param d a dart of the volume to fill
	 */
	virtual void fillHole(Dart d) ;
	//@}

	/*! @name Topological Operators
	 *  Topological operations on 3-G-maps
	 *************************************************************************/

	//@{
	//! Delete the vertex of d
	/*! All the volumes around the vertex are merged into one volume
	 *  @param d a dart of the vertex to delete
	 *  @return a Dart of the resulting volume
	 */
	virtual Dart deleteVertex(Dart d);

	//! Cut the edge of d (all darts around edge orbit are cut)
	/*! @param d a dart of the edge to cut
	 *  @return a dart of the new vertex
	 */
	virtual Dart cutEdge(Dart d);

	//! Uncut the edge of d (all darts around edge orbit are uncut)
	/*! @param d a dart of the edge to uncut
	 */
	virtual bool uncutEdge(Dart d);

	/**
	 * Precondition for deleting edge
	 */
	bool deleteEdgePreCond(Dart d);

	//! Delete the edge of d
	/*! All the volumes around the edge are merged into one volume
	 *  @param d a dart of the edge to delete
	 *  @return a Dart of the resulting volume
	 */
	virtual Dart deleteEdge(Dart d);

	/**
	 * Precondition for splitting face
	 */
	bool splitFacePreCond(Dart d, Dart e);

	//! Split a face inserting an edge between two vertices
	/*! \pre Dart d and e should belong to the same face and be distinct
	 *  @param d dart of first vertex
	 *  @param e dart of second vertex
	 */
	virtual void splitFace(Dart d, Dart e);

	//! Sew two oriented volumes along their faces.
	/*! The oriented faces should not be beta3-linked and have the same degree
	 *  @param d a dart of the first volume
	 *  @param e a dart of the second volume
	 *  @param withBoundary: if false, volumes must have beta3 fixed points (only for construction: import/primitives)
	 */
	virtual void sewVolumes(Dart d, Dart e, bool withBoundary = true);

	//! unsew two oriented volumes along their faces.
	/*! @param d a dart of one volume
	 */
	virtual void unsewVolumes(Dart d);

	//! merge to volume sewed by one face
	/*! @param d a dart of common face
	 */
	virtual bool mergeVolumes(Dart d);

	virtual bool mergeVolumes(Dart /*d*/, Dart /*e*/) { assert("use mergeVolumes(d,e) only in dimension 2");return false;}


	//! Split a volume into two volumes along a edge path
	/*! @param vd a vector of darts
	 */
	virtual void splitVolume(std::vector<Dart>& vd);
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

	//! Compute the number of edges of the vertex of d
	/*! @param d a dart
	 */
	unsigned int vertexDegree(Dart d) const;


	//! Check number of edges of the vertex of d with given parameter
	/*! @param d a dart
	 *	@param vd degree to compare with
	 *  @return  negative/null/positive if vertex degree is less/equal/greater than given degree
	 */
	int checkVertexDegree(Dart d, unsigned int vd) const;


	//! Tell if the vertex of d is on the boundary
	/*! @param d a dart
	 */
	virtual bool isBoundaryVertex(Dart d) const;

	//! Test if dart d and e belong to the same oriented edge
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameOrientedEdge(Dart d, Dart e) const;

	//! Test if dart d and e belong to the same edge
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameEdge(Dart d, Dart e) const;

	//! Compute the number of volumes around the edge of d
	/*! @param d a dart
	 */
	unsigned int edgeDegree(Dart d) const;

	/**
	 * tell if the edge of d is on the boundary of the map
	 */
	bool isBoundaryEdge(Dart d) const;

	/**
	 * find the dart of edge that belong to the boundary
	 * return NIL if the edge is not on the boundary
	 */
	Dart findBoundaryFaceOfEdge(Dart d) const;

	//!Test if dart d and e belong to the same oriented face
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameOrientedFace(Dart d, Dart e) const;

	//!Test if dart d and e belong to the same oriented face
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameFace(Dart d, Dart e) const;

	//! Test if the face is on the boundary
	/*! @param d a dart from the face
	 */
	bool isBoundaryFace(Dart d) const;

	//! Tell if a face of the volume is on the boundary
	/*  @param d a dart
	 */
	bool isBoundaryAdjacentVolume(Dart d) const;

	virtual bool check() const;
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
	* @param d a dart of the edge
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_edge(Dart d, FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on every dart of a face
	/*! @param d a dart of the face
	 *  @param f the functor to apply
	 */
	template <typename FUNC>
	void foreach_dart_of_face(Dart d, FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on every dart of an oriented volume
	/*! @param d a dart of the oriented volume
	 *  @param f the functor to apply
	 */
	template <typename FUNC>
	void foreach_dart_of_oriented_volume(Dart d, FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on every dart of a volume
	/*! @param d a dart of the volume
	 *  @param f the functor to apply
	 */
	template <typename FUNC>
	void foreach_dart_of_volume(Dart d, FUNC& f, unsigned int thread = 0) const;

	/**
	* Apply a functor on each dart of a vertex
	* @param d a dart of the face
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_vertex1(Dart d, FUNC& f, unsigned int thread = 0) const;

	/**
	* Apply a functor on each dart of an edge
	* @param d a dart of the oriented face
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_edge1(Dart d, FUNC& f, unsigned int thread = 0) const;

	/**
	* Apply a functor on each dart of a vertex
	* @param d a dart of the face
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_vertex2(Dart d, FUNC& f, unsigned int thread = 0) const;

	/**
	* Apply a functor on each dart of an edge
	* @param d a dart of the oriented face
	* @param fonct functor obj ref
	*/
	template <typename FUNC>
	void foreach_dart_of_edge2(Dart d, FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on every dart of a face
	/*! @param d a dart of the volume
	 *  @param f the functor to apply
	 */
	template <typename FUNC>
	void foreach_dart_of_face2(Dart d, FUNC& f, unsigned int thread = 0) const;

	/**
	* Apply a functor on each dart of a cc
	* @param d a dart of the cc
	* @param fonct functor obj ref
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

	//! Close a topological hole (a sequence of connected fixed point of phi3). DO NOT USE, only for import/creation algorithm
	/*! \pre dart d MUST be fixed point of phi3 relation
	 *  Add a volume to the map that closes the hole.
	 *  @param d a dart of the hole (with phi3(d)==d)
	 *  @param forboundary tag the created face as boundary (default is true)
	 *  @return the degree of the created volume
	 */
	virtual unsigned int closeHole(Dart d, bool forboundary = true);

	//! Close the map removing topological holes: DO NOT USE, only for import/creation algorithm
	/*! Add volumes to the map that close every existing hole.
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

#include "Topology/gmap/gmap3.hpp"

#endif
