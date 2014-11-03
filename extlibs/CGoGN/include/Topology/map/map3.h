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

#ifndef __MAP3_H__
#define __MAP3_H__

#include "Topology/map/map2.h"

namespace CGoGN
{

/*! \brief The class of dual 3-dimensional combinatorial maps:
 *  set of oriented volumes pairwise sewed by an adjacency relation.
 *  A dual 3-map represents close or open oriented 3-manifolds (volume subdivisions).
 *  - A dual 3-map is made of darts linked by the phi1 permutation
 * 	and/or the phi2 and phi3 one-to-one relation.
 *  - In this class darts are interpreted as oriented edges.
 *  - The phi1 relation defines oriented faces (see tMap1)
 *  and faces may have arbitrary size (degenerated faces are accepted).
 *  - The phi2 relation links oriented faces along oriented edges building
 *  oriented surfaces. A close oriented surface define an oriented volume.
 *  - Volume are linked along whole faces with the phi3 relation 
 *  - Faces that have no phi3-link are border faces. If there exists
 *  such edges the maps is open.
 *  - When every face is phi3-linked, the map is close. In this case
 *  some optimizations are enable that speed up the processing of cells.
 *  @param DART the type of dart used in the class
 */
template <typename MAP_IMPL>
class Map3 : public Map2<MAP_IMPL>
{
protected:
	// protected copy constructor to prevent the copy of map
	Map3(const Map3<MAP_IMPL>& m):Map2<MAP_IMPL>(m) {}

	void init() ;

public:
	typedef MAP_IMPL IMPL;
	typedef Map2<MAP_IMPL> ParentMap;

	inline static unsigned int ORBIT_IN_PARENT(unsigned int o){ return o+7; }
	inline static unsigned int ORBIT_IN_PARENT2(unsigned int o) { return o+5; }

	static const unsigned int IN_PARENT = 7 ;
	static const unsigned int IN_PARENT2 = 5 ;

	static const unsigned int VERTEX_OF_PARENT = VERTEX+7;
	static const unsigned int EDGE_OF_PARENT = EDGE+7;
	static const unsigned int FACE_OF_PARENT = FACE+7;

	static const unsigned int VERTEX_OF_PARENT2 = VERTEX+5;
	static const unsigned int EDGE_OF_PARENT2 = EDGE+5;

	static const unsigned int DIMENSION = 3 ;

	Map3();

	virtual std::string mapTypeName() const;

	virtual unsigned int dimension() const;

	virtual void clear(bool removeAttrib);

	virtual unsigned int getNbInvolutions() const;
	virtual unsigned int getNbPermutations() const;

	/*! @name Basic Topological Operators
	 * Access and Modification
	 *************************************************************************/

	Dart phi3(Dart d) const;

	template <int N>
	Dart phi(Dart d) const;

	Dart alpha0(Dart d) const;

	Dart alpha1(Dart d) const;

	Dart alpha2(Dart d) const;

	Dart alpha_2(Dart d) const;

protected:
	//! Link dart d with dart e by an involution
	/*! @param d,e the darts to link
	 *	- Before:	d->d and e->e
	 *	- After:	d->e and e->d
	 */
	void phi3sew(Dart d, Dart e);

	//! Unlink the current dart by an involution
	/*! @param d the dart to unlink
	 * - Before:	d->e and e->d
	 * - After:		d->d and e->e
	 */
	void phi3unsew(Dart d);

public:
	/*! @name Generator and Deletor
	 *  To generate or delete volumes in a 3-map
	 *************************************************************************/

	//@{
	//! Delete a volume erasing all its darts.
	/*! The phi3-links around the volume are removed
	 *  @param d a dart of the volume
	 */
	virtual void deleteVolume(Dart d, bool withBoundary = true);

	//! Fill a hole with a volume
	/*! \pre Dart d is boundary marked
	 *  @param d a dart of the volume to fill
	 */
	virtual void fillHole(Dart d) ;

	//! Open the mesh Transforming a face in a hole
	/*! \pre Dart d is NOT boundary marked
	 *  @param d a dart of the face filled
	 */
	virtual void createHole(Dart d) ;
	//@}

	/*! @name Topological Operators
	 *  Topological operations on 3-maps
	 *************************************************************************/

	//@{
	//! Split the vertex along a permutation of faces
	/*! \per Darts d & e MUST belong to the same vertex
	 * 	\per Darts d & e MUST belong to different volumes
	 *  \per Works only on the boundary
	 *  @param vd a vector of darts
	 */
	virtual Dart splitVertex(std::vector<Dart>& vd);

	virtual void splitVertex(Dart /*d*/, Dart /*e*/) { assert("use splitVertex(d,e) only in dimension 2");}

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

	//! Collapse an edge (that is deleted) possibly merging its vertices
	/*! \warning This may produce two distinct vertices if the edge
	 *  was the only link between two border faces
	 *  @param d a dart in the deleted edge
	 *  @return a dart of the resulting vertex
	 */
	virtual Dart collapseEdge(Dart d, bool delDegenerateVolumes = true);

	//! Delete a face if and only if it has one or two edges
	/*! If the face is sewed to two distinct adjacent faces,
	 *  then those two faces are sewed
	 *  @param d a dart of the face
	 *  @return true if the collapse has been executed, false otherwise
	 */
//	virtual bool collapseDegeneratedFace(Dart d);

	//! Split Face Pre-condition
	/*!
	 *  @param d dart of first vertex
	 *  @param e dart of second vertex
	 */
	bool splitFacePreCond(Dart d, Dart e);

	//! Split a face inserting an edge between two vertices
	/*! \pre Dart d and e should belong to the same face and be distinct
	 *  @param d dart of first vertex
	 *  @param e dart of second vertex
	 */
	virtual void splitFace(Dart d, Dart e);

	//! Merge the two faces incident to the edge of d.
	/*! Works only for edges of degree 2.
	 *  \warning Darts of the edge of d no longer exist after the call
	 *  @param d a dart in the first face
	 *  @return true if the merge has been executed, false otherwise
	 */
	virtual bool mergeFaces(Dart d);

	//! Collapse a face (that is deleted) possibly merging its vertices
	/*! \warning
	 *  @param d a dart in the deleted face
	 *  @return a dart of the resulting vertex
	 */
	virtual Dart collapseFace(Dart d, bool delDegenerateVolumes = true);

	//! Delete a volume if and only if it has a face with degree < 3 or only 3 vertices
	/*! If the volume is sewed to two distinct adjacent volumes and if the face degree
	 *  of the two adjacent volumes is equal then those two volumes are sewed
	 *  @param d a dart of the face
	 *  @return true if the collapse has been executed, false otherwise
	 */
	bool collapseDegeneretedVolume(Dart d);

	//!! sewVolumes Pre-condition
	bool sewVolumesPreCond(Dart d, Dart e);

	//! Sew two oriented volumes along their faces. 
	/*! The oriented faces should not be phi3-linked and have the same degree
	 *  @param d a dart of the first volume
	 *  @param e a dart of the second volume
	 *  @param withBoundary: if false, volumes must have phi3 fixed points (only for construction: import/primitives)
	 */
	virtual void sewVolumes(Dart d, Dart e, bool withBoundary = true);

	//! Unsew volumes pre-condition
	bool unsewVolumesPreCond(Dart d);

	//! Unsew two oriented volumes along their faces.
	/*! @param d a dart of one volume
	 */
	virtual void unsewVolumes(Dart d, bool withBoundary = true);

	//! Merge two volumes along their common oriented face
	/*! @param d a dart of common face
	 */
	virtual bool mergeVolumes(Dart d, bool deleteFace = true);

	virtual bool mergeVolumes(Dart /*d*/, Dart /*e*/) { assert("use mergeVolumes(d,e) only in dimension 2");return false;}

	//! Split a volume into two volumes along a edge path
	/*! @param vd a vector of darts
	 */
	virtual void splitVolume(std::vector<Dart>& vd);

	//! Split a volume into two volumes along a edge path and add the given face between
	virtual void splitVolumeWithFace(std::vector<Dart>& vd, Dart d);

	//! Collapse a volume (that is deleted) possibly merging its vertices
	/*! \warning
	 *  @param d a dart in the deleted volume
	 *  @return a dart of the resulting vertex
	 */
	virtual Dart collapseVolume(Dart d, bool delDegenerateVolumes = true);
	//@}

    //BROUILLON
    Dart faceToEdge(Dart d);

	/*! @name Topological Queries
	 *  Return or set various topological information
	 *************************************************************************/

	//@{
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

	//! Compute the number of edges of the vertex of d on the boundary
	/*!	@param d a dart
	 */
	unsigned int vertexDegreeOnBoundary(Dart d) const;

	//! Tell if the vertex of d is on the boundary
	/*! @param d a dart
	 */
    bool isBoundaryVertex(Dart d) const;

	//! Find the dart of vertex that belong to the boundary
	/*! return NIL if the vertex is not on the boundary
	 *  @param d a dart
	 */
	Dart findBoundaryFaceOfVertex(Dart d) const;

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

	//! Tell if the edge of d is on the boundary of the map
	/*! @param d a dart
	 */
	bool isBoundaryEdge(Dart d) const;

	//! Find the dart of edge that belong to the boundary
	/*! return NIL if the edge is not on the boundary
	 *  @param d a dart
	 */
	Dart findBoundaryFaceOfEdge(Dart d) const;

	//! Test if dart d and e belong to the same oriented face
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
	bool isVolumeIncidentToBoundary(Dart d) const;

	//! Tell if an edge of the volume is on the boundary
	/*	@param d a dart
	 */
	bool hasBoundaryEdge(Dart d) const;

	//! Check the map completeness
	/*! Test if phi3 and phi2 ares involutions and if phi1 is a permutation
	 */
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
//    template <unsigned int ORBIT, typename FUNC>
//    void foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f, unsigned int thread = 0) const ;
    template <unsigned int ORBIT, typename FUNC>
    void foreach_dart_of_orbit(Cell<ORBIT> c, const FUNC& f, unsigned int thread = 0) const ;

	//! Apply a functor on each dart of a vertex
	/*! @param d a dart of the vertex
	 *  @param fonct the functor
	 */
	template <typename FUNC>
    void foreach_dart_of_vertex(Dart d, const FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on each dart of an edge
	/*! @param d a dart of the oriented edge
	 *  @param fonct the functor
	 */
	template <typename FUNC>
    void foreach_dart_of_edge(Dart d, const FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on each dart of a face
	/*! @param d a dart of the oriented face
	 *  @param fonct the functor
	 */
	template <typename FUNC>
    void foreach_dart_of_face(Dart d, const FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on each dart of a face
	/*! @param d a dart of the oriented face
	 *  @param fonct the functor
	 */
	template <typename FUNC>
    void foreach_dart_of_volume(Dart d, const FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on each dart of a vertex
	/*! @param d a dart of the vertex
	 *  @param fonct the functor
	 */
	template <typename FUNC>
    void foreach_dart_of_vertex1(Dart d, const FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on each dart of an edge
	/*! @param d a dart of the oriented edge
	 *  @param fonct the functor
	 */
	template <typename FUNC>
    void foreach_dart_of_edge1(Dart d, const FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on each dart of a vertex
	/*! @param d a dart of the vertex
	 *  @param fonct the functor
	 */
	template <typename FUNC>
    void foreach_dart_of_vertex2(Dart d, const FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on each dart of an edge
	/*! @param d a dart of the oriented edge
	 *  @param fonct the functor
	 */
	template <typename FUNC>
    void foreach_dart_of_edge2(Dart d, const FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on each dart of an oriented face
	/*! @param d a dart of the oriented face
	 *  @param fonct the functor
	 */
	template <typename FUNC>
    void foreach_dart_of_face2(Dart d, const FUNC& f, unsigned int thread = 0) const;

	//! Apply a functor on each dart of a cc
	/*! @param d a dart of the cc
	 *  @param fonct the functor
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
	//! Reverse the orientation of the map
	/*!
	 */
	void reverseOrientation();

	//! Dual mesh computation
	/*!
	 */
	void computeDual();

	//TODO crade a virer (espece d'extrud)
	// Prend un brin d'une 2-carte
	// - stocke 1 brin par face
	// - decoud chaque face
	// - triangule chaque face
	// - ferme par phi3 chaque volume
	// - recoud le tout
	Dart explodBorderTopo(Dart d);

	void computeDualTest();
	//@}

	/**
	 * @brief move all data from a map2 in a map3
	 * @param mapf the input map2 (which will be empty after)
	 */
	void moveFrom(Map2<MAP_IMPL>& mapf);

};

} // namespace CGoGN

#include "Topology/map/map3.hpp"

#endif
