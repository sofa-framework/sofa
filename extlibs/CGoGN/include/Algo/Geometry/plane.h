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

#ifndef __ALGO_GEOMETRY_PLANE_H__
#define __ALGO_GEOMETRY_PLANE_H__

#include "Geometry/basic.h"
#include "Algo/Geometry/normal.h"
#include "Geometry/plane_3d.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP>
Geom::Plane3D<typename PFP::REAL> trianglePlane(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
	typename PFP::VEC3 n = triangleNormal<PFP>(map, f, position) ;
	return Geom::Plane3D<typename PFP::REAL>(n, position[f.dart]) ;
}

template <typename PFP>
Geom::Plane3D<typename PFP::REAL> facePlane(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
	typename PFP::VEC3 n = faceNormal<PFP>(map, f, position) ;
	return Geom::Plane3D<typename PFP::REAL>(n, position[f.dart]) ;
}

template <typename PFP>
Geom::Plane3D<typename PFP::REAL> vertexTangentPlane(typename PFP::MAP& map, Vertex v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
	typename PFP::VEC3 n = vertexNormal<PFP>(map, v, position) ;
	return Geom::Plane3D<typename PFP::REAL>(n, position[v]) ;
}

} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif
