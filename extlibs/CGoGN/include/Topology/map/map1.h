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

#ifndef __MAP1_H__
#define __MAP1_H__

#include "Topology/generic/mapCommon.h"
#include "Topology/generic/dartmarker.h"
#include "Topology/generic/cellmarker.h"
#include "Algo/Topo/basic.h"

namespace CGoGN
{

/*! \brief The class of dual 1-dimensional combinatorial maps: set of oriented faces.
 *  - A dual 1-map is made of darts linked by the phi1 permutation.
 *  - In this class darts are interpreted as oriented edges.
 *  - The phi1 relation defines cycles of darts or (oriented) faces.
 *  - Faces may have arbitrary size.
 *  - Faces with only one edge (sometime called loops) are accepted.
 *  - Degenerated faces with only two edges are accepted.
 */
template <typename MAP_IMPL>
class Map1 : public MapCommon<MAP_IMPL>
{
protected:
	// protected copy constructor to prevent the copy of map
	Map1(const Map1<MAP_IMPL>& m):MapCommon<MAP_IMPL>(m) {}

	void init() ;

public:
	typedef MAP_IMPL IMPL;
	typedef MapCommon<MAP_IMPL> ParentMap;

	Map1();

	static const unsigned int DIMENSION = 1 ;

	virtual std::string mapTypeName() const;

	virtual unsigned int dimension() const;

	virtual void clear(bool removeAttrib);

	virtual unsigned int getNbInvolutions() const;
	virtual unsigned int getNbPermutations() const;

	/*! @name Basic Topological Operators
	 * Access and Modification
	 *************************************************************************/

	Dart phi1(Dart d) const;

	Dart phi_1(Dart d) const;

	template <int N>
	Dart phi(Dart d) const;

	Dart alpha1(Dart d) const;

	Dart alpha_1(Dart d) const;

protected:
	//! Link the current dart to dart d with a permutation
	/*! @param d the dart to which the current is linked
	 * - Before:	d->f and e->g
	 * - After:		d->g and e->f
	 * Join the permutations cycles of dart d and e
	 * - Starting from two cycles : d->f->...->d and e->g->...->e
	 * - It makes one cycle d->g->...->e->f->...->d
	 * If e = g then insert e in the cycle of d : d->e->f->...->d
	 */
	void phi1sew(Dart d, Dart e);

	//! Unlink the successor of a given dart in a permutation
	/*!	@param d a dart
	 * - Before:	d->e->f
	 * - After:		d->f and e->e
	 */
	void phi1unsew(Dart d);

public:
	/*! @name Generator and Deletor
	 *  To generate or delete faces in a 1-map
	 *************************************************************************/

	//@{
	//! Create an new face made of nbEdges linked darts.
	/*! @param nbEdges the number of edges
	 *  @return return a dart of the face
	 */
	Dart newCycle(unsigned int nbEdges) ;

	//! Delete an oriented face erasing all its darts
	/*! @param d a dart of the face
	 */
	void deleteCycle(Dart d) ;
	//@}

	/*! @name Topological Operators
	 *  Topological operations on 1-maps
	 *************************************************************************/

	//@{
	//! Cut an edge inserting a new dart between d and its successor in the cycle
	/*! @param d the edge to cut
	 * \image hmtl map1_cutEdge.png
	 */
	Dart cutEdge(Dart d);

	//! Undo the cut of the edge of d
	/*! @param d a dart of the edge to uncut
	 */
	void uncutEdge(Dart d);

	//! Collapse an edge of a cycle
	/*!  \warning Dart d no longer exists after the call
	 *  @param d the edge
	 */
	void collapseEdge(Dart d);

	//! Split a cycle between vertices d and e
	/*! \pre Dart d and e MUST be different and belong to the same face
	 *  @param d first dart in the face
	 *  @param e second dart in the face
	 */
	void splitCycle(Dart d, Dart e);

	//! Merge two cycles on vertices d and e
	/*! \pre Dart d and e MUST belong to distinct faces
	 *  @param d a dart in the first face
	 *  @param e a dart in the second face
	 */
	void mergeCycles(Dart d, Dart e);

	//! Link two cycles by adding an edge between two vertices
	/*! \pre Dart d and e MUST be different and belong to distinct face
	 *  @param d first dart in the face
	 *  @param e second dart in the face
	 */
	void linkCycles(Dart d, Dart e);

	//! reverse a face (phi1 become phi_1 and ...)
	/*! @param d a dart of face
	 */
	void reverseCycle(Dart d) ;
	//@}

	/*! @name Topological Queries
	 *  Return or set various topological information
	 *************************************************************************/

	//@{
	//! Test if darts d and e belong to the same cycle
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameCycle(Dart d, Dart e) const;

	//! Length of a cycle (its number of oriented edges)
	/*! @param d a dart of the cycle
	 *  @return the length of the cycle
	 */
	unsigned int cycleDegree(Dart d) const;

	//! Check the Length of a cycle (its number of oriented edges)
	/*! @param d a dart of the cycle
	 *  @param degree the length to compare
	 *  @return  negative/null/positive if face degree is less/equal/greater than given degree
	 */
	 int checkCycleDegree(Dart d, unsigned int degree) const;

	/**
	 * check if the cycle of d is a triangle
	 * @return a boolean indicating if the cycle is a triangle
	 */
	bool isCycleTriangle(Dart d) const;
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
//	template <unsigned int ORBIT, typename FUNC>
//	void foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f, unsigned int thread = 0) const ;

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

	//! Apply a functor on every dart of a connected component
	/*! @param d a dart of the connected component
	 *  @param f the functor to apply
	 */
	template <typename FUNC>
    void foreach_dart_of_cc(Dart d, const FUNC& f, unsigned int thread = 0) const;
	//@}
} ;

} // namespace CGoGN

#include "Topology/map/map1.hpp"

#endif
