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

#include "Topology/generic/traversorCell.h"
#include "Topology/generic/traversor2.h"
#include "Algo/Geometry/basic.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP, typename ATTR_TYPE>
ATTR_TYPE computeLaplacianTopoVertex(
	typename PFP::MAP& map,
	Dart d,
	const VertexAttribute<ATTR_TYPE>& attr)
{
	ATTR_TYPE l(0) ;
	ATTR_TYPE value = attr[d] ;
	unsigned int wSum = 0 ;

	Traversor2VE<typename PFP::MAP> t(map, d) ;
	for(Dart it = t.begin(); it != t.end(); it = t.next())
	{
		l += attr[map.phi1(it)] - value ;
		++wSum ;
	}

	l /= wSum ;
	return l ;
}

template <typename PFP, typename ATTR_TYPE>
ATTR_TYPE computeLaplacianCotanVertex(
	typename PFP::MAP& map,
	Dart d,
	const EdgeAttribute<typename PFP::REAL>& edgeWeight,
	const VertexAttribute<typename PFP::REAL>& vertexArea,
	const VertexAttribute<ATTR_TYPE>& attr)
{
	ATTR_TYPE l(0) ;
	typename PFP::REAL vArea = vertexArea[d] ;
	ATTR_TYPE value = attr[d] ;
	typename PFP::REAL wSum = 0 ;

	Traversor2VE<typename PFP::MAP> t(map, d) ;
	for(Dart it = t.begin(); it != t.end(); it = t.next())
	{
		typename PFP::REAL w = edgeWeight[it] / vArea ;
		l += (attr[map.phi1(it)] - value) * w ;
		wSum += w ;
	}

	l /= wSum ;
	return l ;
}

template <typename PFP, typename ATTR_TYPE>
void computeLaplacianTopoVertices(
	typename PFP::MAP& map,
	const VertexAttribute<ATTR_TYPE>& attr,
	VertexAttribute<ATTR_TYPE>& laplacian)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		laplacian[d] = computeLaplacianTopoVertex<PFP, ATTR_TYPE>(map, d, attr) ;
}

template <typename PFP, typename ATTR_TYPE>
void computeLaplacianCotanVertices(
	typename PFP::MAP& map,
	const EdgeAttribute<typename PFP::REAL>& edgeWeight,
	const VertexAttribute<typename PFP::REAL>& vertexArea,
	const VertexAttribute<ATTR_TYPE>& attr,
	VertexAttribute<ATTR_TYPE>& laplacian)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		laplacian[d] = computeLaplacianCotanVertex<PFP, ATTR_TYPE>(map, d, edgeWeight, vertexArea, attr) ;
}

template <typename PFP>
typename PFP::REAL computeCotanWeightEdge(
	typename PFP::MAP& map,
	Dart d,
	const VertexAttribute<typename PFP::VEC3>& position)
{
	if(map.isBoundaryEdge(d))
	{
		const typename PFP::VEC3& p1 = position[d] ;
		const typename PFP::VEC3& p2 = position[map.phi1(d)] ;
		const typename PFP::VEC3& p3 = position[map.phi_1(d)] ;

		typename PFP::REAL cot_alpha = 1 / tan(Geom::angle(p1 - p3, p2 - p3)) ;
		return 0.5 * cot_alpha ;
	}
	else
	{
		const typename PFP::VEC3& p1 = position[d] ;
		const typename PFP::VEC3& p2 = position[map.phi1(d)] ;
		const typename PFP::VEC3& p3 = position[map.phi_1(d)] ;
		const typename PFP::VEC3& p4 = position[map.phi_1(map.phi2(d))] ;

		typename PFP::REAL cot_alpha = 1 / tan(Geom::angle(p1 - p3, p2 - p3)) ;
		typename PFP::REAL cot_beta = 1 / tan(Geom::angle(p2 - p4, p1 - p4)) ;
		return 0.5 * ( cot_alpha + cot_beta ) ;
	}
}

template <typename PFP>
void computeCotanWeightEdges(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3>& position,
	EdgeAttribute<typename PFP::REAL>& edgeWeight)
{
	TraversorE<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		edgeWeight[d] = computeCotanWeightEdge<PFP>(map, d, position) ;
}

} // namespace Geometry

} // namespace Surface


namespace Volume
{

namespace Geometry
{

template <typename PFP, typename ATTR_TYPE>
ATTR_TYPE computeLaplacianTopoVertex(typename PFP::MAP& map, Dart d, const VertexAttribute<ATTR_TYPE>& attr)
{
	ATTR_TYPE l(0) ;
	ATTR_TYPE value = attr[d] ;
	unsigned int wSum = 0 ;

	Traversor3VE<typename PFP::MAP> t(map, d) ;
	for(Dart it = t.begin(); it != t.end(); it = t.next())
	{
		l += attr[map.phi1(it)] - value ;
		++wSum ;
	}

	l /= wSum ;
	return l ;
}

template <typename PFP, typename ATTR_TYPE>
void computeLaplacianTopoVertices(typename PFP::MAP& map, const VertexAttribute<ATTR_TYPE>& attr,
									VertexAttribute<ATTR_TYPE>& laplacian)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		laplacian[d] = computeLaplacianTopoVertex<PFP, ATTR_TYPE>(map, d, attr) ;
}


} // namespace Geometry

} // namespace Volume


} // namespace Algo

} // namespace CGoGN
