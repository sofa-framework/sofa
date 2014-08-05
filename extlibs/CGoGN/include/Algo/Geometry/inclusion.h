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

#ifndef __ALGO_GEOMETRY_INCLUSION_H_
#define __ALGO_GEOMETRY_INCLUSION_H_

#include <vector>
#include "Geometry/basic.h"
#include "Topology/generic/dart.h"
#include "Geometry/inclusion.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

/**
 * test if the volume is convex
 * @param the map
 * @param a volume
 * @param true if the faces of the volume must be in CCW order (default=true)
 */
template <typename PFP>
bool isConvex(typename PFP::MAP& map, Vol v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, bool CCW, unsigned int thread = 0);

/**
 * test if a point is inside a volume
 * @param map the map
 * @param a volume
 * @param the point
 */
template <typename PFP>
bool isPointInVolume(typename PFP::MAP& map, Vol v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const typename PFP::VEC3& point);

/**
 * test if a point is inside a volume
 * @param map the map
 * @param a convex volume
 * @param the point
 */
template <typename PFP>
bool isPointInConvexVolume(typename PFP::MAP& map, Vol v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const typename PFP::VEC3& point, bool CCW = true);

/**
 * test if a point is inside a face
 * @param map the map
 * @param a face
 * @param the point
 */
template <typename PFP>
bool isPointInConvexFace(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const typename PFP::VEC3& point, bool CCW);

/**
 * test if a point is inside a face in a plane
 * @param map the map
 * @param a face
 * @param the point
 */
template <typename PFP>
bool isPointInConvexFace2D(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const typename PFP::VEC3& point, bool CCW = true);

/**
 * test if a point is on an edge
 * @param map the map
 * @param an edge
 * @param the point
 */
template <typename PFP>
bool isPointOnEdge(typename PFP::MAP& map, Edge e, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const typename PFP::VEC3& point);

/**
 * test if a point is on an half-edge defined by a dart
 * @param map the map
 * @param a Dart
 * @param the point
 */
template <typename PFP>
bool isPointOnHalfEdge(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const typename PFP::VEC3& point);

/**
 * test if a point is on a vertex
 * @param map the map
 * @param a vertex
 * @param the point
 */
template <typename PFP>
bool isPointOnVertex(typename PFP::MAP& map, Vertex v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const typename PFP::VEC3& point);

/**
 * test if a face is intersecting or totally included in a tetrahedron
 * TODO to test
 * @param map the map
 * @param f a face
 * @param the points of the tetra (0,1,2) the first face and (3) the last point of the tetra, in well oriented order
 * @param true if the faces of the tetra are in CCW order (default=true)
 */
template <typename PFP>
bool isConvexFaceInOrIntersectingTetrahedron(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const typename PFP::VEC3 points[4], bool CCW);

} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Geometry/inclusion.hpp"

#endif
