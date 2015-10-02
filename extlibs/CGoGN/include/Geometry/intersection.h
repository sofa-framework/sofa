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

#ifndef __INTERSECTION__
#define __INTERSECTION__

#include "Geometry/basic.h"
#include "Geometry/inclusion.h"
#include <limits>

namespace CGoGN
{

namespace Geom
{

enum Intersection
{
	NO_INTERSECTION = 0,
	VERTEX_INTERSECTION = 1,
	EDGE_INTERSECTION = 2,
	FACE_INTERSECTION = 3
} ;

/**
 * test the intersection between a line and a triangle
 * @param P a point on the line
 * @param Dir line direction
 * @param PlaneP point of plane
 * @param NormP normal of plane
 * @param Inter store the intersection point
 * @return the intersection ( FACE_INTERSECTION = OK, EDGE_INTERSECTION = line inside of plane)
 */
template <typename VEC3>
Intersection intersectionLinePlane(const VEC3& P, const VEC3& Dir, const VEC3& PlaneP, const VEC3& NormP, VEC3& Inter) ;

/**
 * test the intersection between a line and a triangle
 * @param P a point on the line
 * @param Dir line direction
 * @param Plane plane
 * @param Inter store the intersection point
 * @return the intersection ( FACE_INTERSECTION = OK, EDGE_INTERSECTION = line inside of plane)
 */
template <typename VEC3, typename PLANE>
Intersection intersectionLinePlane(const VEC3& P, const VEC3& Dir, const PLANE& Plane, VEC3& Inter) ;

/**
 * test the intersection between a ray and a triangle (optimized version with triple product)
 * @param P a point on the line
 * @param Dir line direction
 * @param Ta triangle point 1
 * @param Tb triangle point 2
 * @param Tc triangle point 3
 * @param Inter store the intersection point
 * @return the intersection ( FACE_INTERSECTION / EDGE_INTERSECTION / VERTEX_INTERSECTION / NO_INTERSECTION)
 */
template <typename VEC3>
Intersection intersectionRayTriangle(const VEC3& P, const VEC3& Dir, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, VEC3& Inter) ;

/**
 * test the intersection between a ray and a triangle (optimized version with triple product)
 * @param P a point on the line
 * @param Dir line direction
 * @param Ta triangle point 1
 * @param Tb triangle point 2
 * @param Tc triangle point 3
 * @param Inter store the intersection point
 * @return the intersection ( FACE_INTERSECTION / EDGE_INTERSECTION / VERTEX_INTERSECTION / NO_INTERSECTION)
 */
template <typename VEC3>
Intersection intersectionRayTriangleOpt(const VEC3& P, const VEC3& Dir, const VEC3& Ta,  const VEC3& Tb, const VEC3& Tc, VEC3& Inter);

/**
 * test the intersection between a ray and a triangle (optimized version with triple product)
 * @param P a point on the line
 * @param Dir line direction
 * @param Ta triangle point 1
 * @param Tb triangle point 2
 * @param Tc triangle point 3
 * @return the intersection ( FACE_INTERSECTION / EDGE_INTERSECTION / VERTEX_INTERSECTION / NO_INTERSECTION)
 */
template <typename VEC3>
Intersection intersectionRayTriangleOpt(const VEC3& P, const VEC3& Dir, const VEC3& Ta,  const VEC3& Tb, const VEC3& Tc);

/**
 * test the intersection between a ray and a triangle, but test only face intersection and
 * @param P a point on the line
 * @param Dir line direction
 * @param Ta triangle point 1
 * @param Tb triangle point 2
 * @param Tc triangle point 3
 * @param Inter store the intersection point
 * @return the intersection ( FACE_INTERSECTION / EDGE_INTERSECTION / VERTEX_INTERSECTION / NO_INTERSECTION)
 */
//template <typename VEC3>
//Intersection intersectionRayTriangleFaceOnly(const VEC3& P, const VEC3& Dir, const VEC3& Ta,  const VEC3& Tb, const VEC3& Tc)

/**
 * test the intersection between a line and a triangle
 * @param P a point on the line
 * @param Dir line direction
 * @param Ta triangle point 1
 * @param Tb triangle point 2
 * @param Tc triangle point 3
 * @param Inter store the intersection point
 * @return the intersection
 */
template <typename VEC3>
Intersection intersectionLineTriangle(const VEC3& P, const VEC3& Dir, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, VEC3& Inter) ;

/**
 * test the intersection between a line and a triangle the line MUST be in the plane of the triangle, assumed to be CCW
 * TODO : complete edge ray intersection
 * @param P a point on the line
 * @param Dir a direction for the line
 * @param Ta triangle point 1
 * @param Tb triangle point 2
 * @param Tc triangle point 3
 * @param Inter store the intersection point
 * @return the intersection
 */
template <typename VEC3>
Intersection intersectionLineTriangle2D(const VEC3& P, const VEC3& Dir, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, VEC3& Inter) ;

/**
 * test the intersection between a segment and a triangle
 * @param PA segment point 1
 * @param PB segment point 2
 * @param Ta triangle point 1
 * @param Tb triangle point 2
 * @param Tc triangle point 3
 * @param Inter store the intersection point
 * @return the intersection
 */
template <typename VEC3>
Intersection intersectionSegmentTriangle(const VEC3& PA, const VEC3& PB, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, VEC3& Inter, double epsilon) ;

template <typename VEC3, typename PLANE3D>
Intersection intersectionPlaneRay(const PLANE3D& pl,const VEC3& p1,const VEC3& dir, VEC3& Inter) ;

template <typename VEC3>
Intersection intersection2DSegmentSegment(const VEC3& PA, const VEC3& PB, const VEC3& QA, const VEC3& QB, VEC3& Inter) ;

template <typename VEC3>
Intersection intersectionSegmentPlan(const VEC3& PA, const VEC3& PB, const VEC3& PlaneP, const VEC3& NormP); //, VEC3& Inter) ;

template <typename VEC3>
Intersection intersectionSegmentPlan(const VEC3& PA, const VEC3& PB, const VEC3& PlaneP, const VEC3& NormP, VEC3& Inter) ;


template <typename VEC3>
Intersection intersectionTrianglePlan(const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, const VEC3& PlaneP, const VEC3& NormP);//, VEC3& Inter) ;

template <typename VEC3>
Intersection intersectionSegmentHalfPlan(const VEC3& PA, const VEC3& PB,
		const VEC3& P, const VEC3& DirP, const VEC3& OrientP);//, VEC3& Inter)

template <typename VEC3>
Intersection intersectionTriangleHalfPlan(const VEC3& Ta, const VEC3& Tb, const VEC3& Tc,
		const VEC3& P, const VEC3& DirP, const VEC3& OrientP); //, VEC3& Inter) ;


/**
* compute intersection between line and segment
* @param A point of line
* @param AB vector of line
* @param AB2 AB*AB (for optimization if call several times with AB
* @param P first point of segment
* @param Q second point of segment
* @param inter the computed intersection
* @return intersection of not
*/
template <typename VEC3>
bool interLineSeg(const VEC3& A, const VEC3& AB, typename VEC3::DATA_TYPE AB2,
				  const VEC3& P, const VEC3& Q, VEC3& inter);
} // namespace Geom

} // namespace CGoGN

#include "intersection.hpp"

#endif
