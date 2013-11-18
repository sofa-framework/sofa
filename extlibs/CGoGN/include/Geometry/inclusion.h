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

#ifndef __INCLUSION__
#define __INCLUSION__


#include "Geometry/basic.h"
#include "Geometry/orientation.h"

namespace CGoGN
{

namespace Geom
{

enum Inclusion
{
	NO_INCLUSION,
	VERTEX_INCLUSION,
	EDGE_INCLUSION,
	FACE_INCLUSION
} ;

/**
 * test if a point is inside a triangle, the point MUST be in the plane of the triangle
 * @param point the point
 * @param Ta triangle point 1
 * @param Tb triangle point 2
 * @param Tc triangle point 3
 * @return the inclusion
 */
template <typename VEC3>
Inclusion isPointInTriangle(const VEC3& point, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc) ;

template <typename VEC3>
bool isPointInSphere(const VEC3& point, const VEC3& center, const typename VEC3::DATA_TYPE& radius) ;

/**
 * test if a segment is inside a triangle, the segment MUST be in the plane of the triangle
 * TODO to test
 * @param the point
 * @param Ta triangle point 1
 * @param Tb triangle point 2
 * @param Tc triangle point 3
 * @param N the normal of the triangle
 */
template <typename VEC3>
Inclusion isSegmentInTriangle(const VEC3& P1, const VEC3& P2, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, const VEC3& N) ;

/**
 * test if a point is inside a tetrahedron
 * @param the points of the tetra (0,1,2) the first face and (3) the last point of the tetra, in well oriented order
 * @param the point
 * @return true if the faces of the tetra are in CCW order (default=true)
 */
template <typename VEC3>
bool isPointInTetrahedron(VEC3 points[4], VEC3& point, bool CCW) ;

/**
 * test if an edge is inside or intersect a tetrahedron
 * @param the points of the tetra (0,1,2) the first face and (3) the last point of the tetra, in well oriented order
 * @param point1 a vertex of the edge
 * @param point2 the other vertex of the edge
 * @return true if the faces of the tetra are in CCW order (default=true)
 */
template <typename VEC3>
bool isEdgeInOrIntersectingTetrahedron(VEC3 points[4], VEC3& point1, VEC3& point2, bool CCW) ;

/**
 * test if two points are equals (with an epsilon)
 * @param point1 first point
 * @param point2 second point
 * @return true if the points are equals
 */
template <typename VEC3>
bool arePointsEquals(const VEC3& point1,const VEC3& point2) ;

} // namespace Geom

} // namespace CGoGN

#include "inclusion.hpp"

#endif
