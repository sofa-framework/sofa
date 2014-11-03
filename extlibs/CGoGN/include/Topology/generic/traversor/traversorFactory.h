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

#ifndef __TRAVERSORFACTORY_H__
#define __TRAVERSORFACTORY_H__
#include "traversorGen.h"
namespace CGoGN
{

template <typename MAP>
class TraversorFactory
{
public:

	/**
	 * Factory of incident traversors creation
	 * @param map the map in which we work
	 * @param dart the initial dart of traversal
	 * @param dim the dimension of traversal (2 or 3)
	 * @param orbX incident from cell
	 * @param orbY incident to cell
	 * @return a ptr on Generic Traversor
	 */
	static Traversor* createIncident(MAP& map, Dart dart, unsigned int dim, unsigned int orbX, unsigned int orbY);

	/**
	 * Factory of adjacent traversors creation
	 * @param map the map in which we work
	 * @param dart the initial dart of traversal
	 * @param dim the dimension of traversal (2 or 3)
	 * @param orbX incident from cell
	 * @param orbY incident to cell
	 * @return a ptr on Generic Traversor
	 */
	static Traversor* createAdjacent(MAP& map, Dart dart, unsigned int dim, unsigned int orbX, unsigned int orbY);

	/**
	 * Factory of darts of orbit traversors creation
	 * @param map the map in which we work
	 * @param dart the initial dart of traversal
	 * @param orb the orbit
	 * @return a ptr on Generic Traversor
	 */
	static Traversor* createDartsOfOrbits(MAP& map, Dart dart, unsigned int orb);

	/**
	 * Factory of incident traversors creation
	 * @param map the map in which we work
	 * @param forceDartMarker (default value false)
	 * @param thread (default value 0)
	 * @return a ptr on Generic Traversor
	 */
	static Traversor* createCell(MAP& map, unsigned int orb, bool forceDartMarker = false, unsigned int thread = 0);
};

} // namespace CGoGN


#include "Topology/generic/traversor/traversorFactory.hpp"

#endif

