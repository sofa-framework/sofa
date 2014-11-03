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

#ifndef __ALGO_TOPO_EMBEDDING__
#define __ALGO_TOPO_EMBEDDING__
#include <boost/lambda/lambda.hpp>
#include <boost/bind.hpp>
namespace bl = boost::lambda;
namespace CGoGN
{

namespace Algo
{

namespace Topo
{

/**
* Set the index of the associated cell to all the darts of an orbit
* @param orbit orbit to embed
* @param d a dart of the topological vertex
* @param em index of attribute to store as embedding
*/
template <unsigned int ORBIT, typename MAP>
inline void setOrbitEmbedding(MAP& m, Cell<ORBIT> c, unsigned int em)
{
	assert(m.template isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");
    assert(em != EMBNULL);
//    std::cerr << "setOrbitEmbedding called on a " << ORBIT << "-cell." << " em = " << em << std::endl;
//	m.foreach_dart_of_orbit(c, [&] (Dart d) { m.template setDartEmbedding<ORBIT>(d, em); });
    m. template foreach_dart_of_orbit<ORBIT>(c, (bl::bind(&MAP::template setDartEmbedding<ORBIT>, boost::ref(m), bl::_1, boost::cref(em) ))) ;
}

/**
 * Set the index of the associated cell to all the darts of an orbit
 * !!! WARNING !!! use only on freshly inserted darts (no unref is done on old embedding)!!! WARNING !!!
 */
template <unsigned int ORBIT, typename MAP>
inline void initOrbitEmbedding(MAP& m, Cell<ORBIT> c, unsigned int em)
{
	assert(m.template isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");
//	m.foreach_dart_of_orbit(c, [&] (Dart d) { m.template initDartEmbedding<ORBIT>(d, em); });
    m.template foreach_dart_of_orbit(c,bl::bind(&MAP::template initDartEmbedding<ORBIT>, boost::ref(m), bl::_1, boost::cref(em) )) ;
}

/**
* Associate an new cell to all darts of an orbit
* @param orbit orbit to embed
* @param d a dart of the topological cell
* @return index of the attribute in table
*/
template <unsigned int ORBIT, typename MAP>
inline unsigned int setOrbitEmbeddingOnNewCell(MAP& m, Cell<ORBIT> c)
{
	assert(m.template isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");
	unsigned int em = m.template newCell<ORBIT>();
	setOrbitEmbedding<ORBIT>(m, c, em);
	return em;
}

/**
 * Associate an new cell to all darts of an orbit
 * !!! WARNING !!! use only on freshly inserted darts (no unref is done on old embedding)!!! WARNING !!!
 */
template <unsigned int ORBIT, typename MAP>
inline unsigned int initOrbitEmbeddingOnNewCell(MAP& m, Cell<ORBIT> d)
{
	assert(m.template isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");

	unsigned int em = m.template newCell<ORBIT>();
	initOrbitEmbedding<ORBIT>(m, d, em);
	return em;
}

/**
 * Copy the cell associated to a dart over an other dart
 * @param orbit attribute orbit to use
 * @param d the dart to overwrite (dest)
 * @param e the dart to copy (src)
 */
template <unsigned int ORBIT, typename MAP>
inline void copyCellAttributes(MAP& m, Cell<ORBIT> d, Cell<ORBIT> e)
{
	assert(m.template isOrbitEmbedded<ORBIT>() || !"Invalid parameter: orbit not embedded");

	unsigned int dE = m.getEmbedding(d) ;
	unsigned int eE = m.getEmbedding(e) ;
	if(eE != EMBNULL)	// if the source is NULL, nothing to copy
	{
		if(dE == EMBNULL)	// if the dest is NULL, create a new cell
			dE = setOrbitEmbeddingOnNewCell(m, d) ;
		AttributeContainer& cont = m.template getAttributeContainer<ORBIT>();
		cont.copyLine(dE, eE) ;	// copy the data
	}
}

template <unsigned int DIM, unsigned int ORBIT, typename MAP>
void boundaryMarkOrbit(MAP& m, Cell<ORBIT> c)
{
//	m.foreach_dart_of_orbit(c, [&] (Dart d) { m.template boundaryMark<DIM>(d); });
    m.foreach_dart_of_orbit(c,bl::bind(&MAP::template boundaryMark<DIM>, boost::ref(m), bl::_1)) ;

}

template <unsigned int DIM, unsigned int ORBIT, typename MAP>
void boundaryUnmarkOrbit(MAP& m, Cell<ORBIT> c)
{
//	m.foreach_dart_of_orbit(c, [&] (Dart d)	{ m.template boundaryUnmark<DIM>(d);	});
    m.foreach_dart_of_orbit(c,bl::bind(&MAP::template boundaryUnmark<DIM>, boost::ref(m), bl::_1)) ;
}

} // namespace Topo

} // namespace Algo

} // namespace CGoGN

#endif
