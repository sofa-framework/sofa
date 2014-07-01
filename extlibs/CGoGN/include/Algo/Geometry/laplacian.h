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

#ifndef __ALGO_GEOMETRY_LAPLACIAN_H__
#define __ALGO_GEOMETRY_LAPLACIAN_H__

#include "Geometry/basic.h"

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
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr) ;

template <typename PFP, typename ATTR_TYPE>
ATTR_TYPE computeLaplacianCotanVertex(
	typename PFP::MAP& map,
	Dart d,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeWeight,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& vertexArea,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr) ;

template <typename PFP, typename ATTR_TYPE>
void computeLaplacianTopoVertices(
	typename PFP::MAP& map,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	VertexAttribute<ATTR_TYPE, typename PFP::MAP>& laplacian) ;

template <typename PFP, typename ATTR_TYPE>
void computeLaplacianCotanVertices(
	typename PFP::MAP& map,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeWeight,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& vertexArea,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	VertexAttribute<ATTR_TYPE, typename PFP::MAP>& laplacian) ;

template <typename PFP>
typename PFP::REAL computeCotanWeightEdge(
	typename PFP::MAP& map,
	Dart d,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position) ;

template <typename PFP>
void computeCotanWeightEdges(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeWeight) ;

} // namespace Geometry

} // namespace Surface

namespace Volume
{

namespace Geometry
{

template <typename PFP, typename ATTR_TYPE>
ATTR_TYPE computeLaplacianTopoVertex(
	typename PFP::MAP& map,
	Dart d,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr) ;

template <typename PFP, typename ATTR_TYPE>
void computeLaplacianTopoVertices(
	typename PFP::MAP& map,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	VertexAttribute<ATTR_TYPE, typename PFP::MAP>& laplacian) ;

} // namespace Geometry

} // namespace Volume

} // namespace Algo

} // namespace CGoGN

#include "Algo/Geometry/laplacian.hpp"

#endif
