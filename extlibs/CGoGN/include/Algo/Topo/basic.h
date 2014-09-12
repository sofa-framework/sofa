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

#ifndef __ALGO_TOPO_BASIC__
#define __ALGO_TOPO_BASIC__

#include "Topology/generic/attributeHandler.h"
#include "Topology/generic/traversor/traversorCell.h"
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/if.hpp>

namespace bl = boost::lambda;

namespace CGoGN
{

namespace Algo
{

namespace Topo
{

template <unsigned int ORBIT, typename MAP>
unsigned int getNbOrbits(const MAP& map)
{
	unsigned int cpt = 0;
    foreach_cell<ORBIT>(map, ++bl::var(cpt) , FORCE_DART_MARKING);
	return cpt;
}

template <typename MAP>
unsigned int getNbOrbits(const MAP& map, unsigned int orbit)
{
	switch(orbit)
	{
		case DART:		return getNbOrbits<DART, MAP>(map);
		case VERTEX: 	return getNbOrbits<VERTEX, MAP>(map);
		case EDGE: 		return getNbOrbits<EDGE, MAP>(map);
		case FACE: 		return getNbOrbits<FACE, MAP>(map);
		case VOLUME: 	return getNbOrbits<VOLUME, MAP>(map);
		case VERTEX1: 	return getNbOrbits<VERTEX1, MAP>(map);
		case EDGE1: 	return getNbOrbits<EDGE1, MAP>(map);
		case VERTEX2: 	return getNbOrbits<VERTEX2, MAP>(map);
		case EDGE2:		return getNbOrbits<EDGE2, MAP>(map);
		case FACE2:		return getNbOrbits<FACE2, MAP>(map);
		default: 		assert(!"Cells of this dimension are not handled"); break;
	}
	return 0;
}

/**
 * Traverse the map and embed all orbits of the given dimension with a new cell
 * @param realloc if true -> all the orbits are embedded on new cells, if false -> already embedded orbits are not impacted
 */
template <unsigned int ORBIT, typename MAP>
void initAllOrbitsEmbedding(MAP& map, bool realloc = false)
{
	if(!map.template isOrbitEmbedded<ORBIT>())
		map.template addEmbedding<ORBIT>() ;
//	foreach_cell<ORBIT>(map, [&] (Cell<ORBIT> c)
//	{
//		if(realloc || map.template getEmbedding<ORBIT>(c) == EMBNULL)
//			Algo::Topo::setOrbitEmbeddingOnNewCell<ORBIT>(map, c) ;
//	});
    foreach_cell<ORBIT>(map, (bl::if_( (realloc || (bl::bind(&MAP::template getEmbedding<ORBIT>, boost::cref(map), bl::_1) == EMBNULL)))[ (bl::bind(&Algo::Topo::template setOrbitEmbeddingOnNewCell<ORBIT, MAP>, boost::ref(map), bl::_1)) ]));
}

/**
 * use the given attribute to store the indices of the cells of the corresponding orbit
 * @return the number of cells of the orbit
 */
template <unsigned int ORBIT, typename MAP>
unsigned int computeIndexCells(MAP& map, AttributeHandler<unsigned int, ORBIT, MAP>& idx)
{
	AttributeContainer& cont = map.template getAttributeContainer<ORBIT>();
	unsigned int cpt = 0 ;
	for (unsigned int i = cont.begin(); i != cont.end(); cont.next(i))
		idx[i] = cpt++ ;
	return cpt ;
}

/**
 * ensure that each embedding is pointed by only one orbit
 */
template <unsigned int ORBIT, typename MAP>
void bijectiveOrbitEmbedding(MAP& map)
{
	if(!map.template isOrbitEmbedded<ORBIT>())
		map.template addEmbedding<ORBIT>() ;

	AttributeHandler<int, ORBIT, MAP> counter = map.template addAttribute<int, ORBIT, MAP>("tmpCounter") ;
	counter.setAllValues(int(0)) ;

//	foreach_cell<ORBIT>(map, [&] (Cell<ORBIT> d)
//	{
//		unsigned int emb = map.template getEmbedding<ORBIT>(d) ;
//		if (emb != EMBNULL)
//		{
//			if (counter[d] > 0)
//			{
//				unsigned int newEmb = Algo::Topo::setOrbitEmbeddingOnNewCell<ORBIT>(map, d) ;
//				map.template copyCell<ORBIT>(newEmb, emb) ;
////				map.template getAttributeContainer<ORBIT>().copyLine(newEmb, emb) ;
//			}
//			counter[d]++ ;
//		}
//	},
//	FORCE_DART_MARKING);
    unsigned int emb = EMBNULL;
    foreach_cell<ORBIT>( (map,boost::ref(emb) = bl::bind(&MAP::template getEmbedding<ORBIT>, boost::cref(map), bl::_1),
                          bl::if_(boost::ref(emb) != EMBNULL)[
                                      (bl::if_(bl::bind(&AttributeHandler<int, ORBIT, MAP>::operator [],boost::cref(counter), bl::_1) > 0)
                                        [ (bl::bind(&MAP::template copyCell<ORBIT>, boost::ref(map), bl::bind(&Algo::Topo::template setOrbitEmbeddingOnNewCell<ORBIT, MAP>, boost::ref(map), bl::_1), boost::ref(emb))) ]
                                      , bl::bind(&AttributeHandler<int, ORBIT, MAP>::operator [],boost::ref(counter), bl::_1)++)
            ]
                          ), FORCE_DART_MARKING);

	map.removeAttribute(counter) ;
}

} // namespace Topo

} // namespace Algo

} // namespace CGoGN

#endif
