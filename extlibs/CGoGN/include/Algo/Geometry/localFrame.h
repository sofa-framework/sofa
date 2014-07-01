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

#ifndef __ALGO_GEOMETRY_LOCALFRAME_H__
#define __ALGO_GEOMETRY_LOCALFRAME_H__

#include "Geometry/basic.h"
#include "Geometry/matrix.h"
#include "Algo/Geometry/basic.h"
#include "Algo/Geometry/normal.h"

namespace CGoGN
{

namespace Algo
{

namespace Geometry
{

// compute a local frame on the vertex of dart d

template <typename PFP>
void vertexLocalFrame(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, typename PFP::VEC3& X, typename PFP::VEC3& Y, typename PFP::VEC3& Z)
{
	Z = Algo::Surface::Geometry::vertexNormal<PFP>(map, d, position) ;
	X = Algo::Surface::Geometry::vectorOutOfDart<PFP>(map, d, position) ;
	Y = Z ^ X ;
	Y.normalize() ;
	X = Y ^ Z ;
	X.normalize() ;
}

template <typename PFP>
typename PFP::MATRIX33 vertexLocalFrame(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
	typename PFP::VEC3 X, Y, Z ;
	vertexLocalFrame<PFP>(map, d, position, X, Y, Z) ;
	typename PFP::MATRIX33 frame ;
	frame(0,0) = X[0] ;	frame(0,1) = X[1] ;	frame(0,2) = X[2] ;
	frame(1,0) = Y[0] ;	frame(1,1) = Y[1] ;	frame(1,2) = Y[2] ;
	frame(2,0) = Z[0] ;	frame(2,1) = Z[1] ;	frame(2,2) = Z[2] ;
	return frame ;
}

// compute a local frame on the vertex of dart d with a prescribed normal vector

template <typename PFP>
void vertexLocalFrame(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, typename PFP::VEC3& normal, typename PFP::VEC3& X, typename PFP::VEC3& Y, typename PFP::VEC3& Z)
{
	Z = normal ;
	X = Algo::Surface::Geometry::vectorOutOfDart<PFP>(map, d, position) ;
	Y = Z ^ X ;
	Y.normalize() ;
	X = Y ^ Z ;
	X.normalize() ;
}

template <typename PFP>
typename PFP::MATRIX33 vertexLocalFrame(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, typename PFP::VEC3& normal)
{
	typename PFP::VEC3 X, Y, Z ;
	vertexLocalFrame<PFP>(map, d, position, normal, X, Y, Z) ;
	typename PFP::MATRIX33 frame ;
	frame(0,0) = X[0] ;	frame(0,1) = X[1] ;	frame(0,2) = X[2] ;
	frame(1,0) = Y[0] ;	frame(1,1) = Y[1] ;	frame(1,2) = Y[2] ;
	frame(2,0) = Z[0] ;	frame(2,1) = Z[1] ;	frame(2,2) = Z[2] ;
	return frame ;
}

} // namespace Geometry

} // namespace Algo

} // namespace CGoGN

#endif
