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

#ifndef __GMAP0_H__
#define __GMAP0_H__

#include "Topology/generic/attribmap.h"
#include "Topology/generic/dartmarker.h"
#include "Topology/generic/cellmarker.h"

namespace CGoGN
{

/**
* The class of 0-GMap
* Warning here we use beta instead of classic alpha
*/
class GMap0 : public AttribMap
{
protected:
	AttributeMultiVector<Dart>* m_beta0 ;

	void init() ;

public:
	GMap0();
	static const unsigned int DIMENSION = 0 ;

	virtual std::string mapTypeName() const;

	virtual unsigned int dimension() const;

	virtual void clear(bool removeAttrib);

	virtual void update_topo_shortcuts();

	virtual void compactTopoRelations(const std::vector<unsigned int>& oldnew);

	/*! @name Basic Topological Operators
	 * Access and Modification
	 *************************************************************************/

	virtual Dart newDart();

	Dart beta0(const Dart d);

	void beta0sew(Dart d, Dart e);

	void beta0unsew(Dart d);

	/*! @name Constructors and Destructors
	 *  To generate or delete cells in a 0-G-map
	 *************************************************************************/

	//@{
	/**
	* create an edge
	* @return a dart of the edge
	*/
	Dart newEdge();

	/**
	* delete an edge
	* @param d a dart of the edge
	*/
	void deleteEdge(Dart d);
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
//	bool foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread = 0);
	//@}
};

} // namespace CGoGN

#include "Topology/gmap/gmap0.hpp"

#endif
