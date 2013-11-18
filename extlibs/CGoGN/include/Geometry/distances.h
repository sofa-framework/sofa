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

#ifndef __DISTANCES__
#define __DISTANCES__

#include "Geometry/basic.h"

namespace CGoGN
{

namespace Geom
{

/**
* compute squared distance from point to triangle plane
* @param P the point
* @param A triangle point 1
* @param B triangle point 2
* @param C triangle point 3
* @return the squared distance
*/
template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistancePoint2TrianglePlane(const VEC3& P, const VEC3& A, const VEC3& B, const VEC3& C) ;

/**
* compute distance from point to triangle plane
* @param P the point
* @param A triangle point 1
* @param B triangle point 2
* @param C triangle point 3
* @return the distance
*/
template <typename VEC3>
typename VEC3::DATA_TYPE distancePoint2TrianglePlane(const VEC3& P, const VEC3& A, const VEC3& B, const VEC3& C) ;

/**
* compute squared distance from point to triangle
* @param P the point
* @param A triangle point 1
* @param B triangle point 2
* @param C triangle point 3
* @return the squared distance
*/
template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistancePoint2Triangle(const VEC3& P, const VEC3& A, const VEC3& B, const VEC3& C) ;

/**
* compute squared distance from point to line
* @param P the point
* @param A a point on the line
* @param AB line direction
* @param AB2 squared norm of vector AB
* @return the squared distance
*/
template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistanceLine2Point(const VEC3& A, const VEC3& AB, typename VEC3::DATA_TYPE AB2, const VEC3& P) ;

/**
* compute squared distance from point to line
* @param P the point
* @param A line point 1
* @param B line point 2
* @return the squared distance
*/
template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistanceLine2Point(const VEC3& A, const VEC3& B, const VEC3& P) ;

/**
* compute squared distance from line to line
* @return the squared distance
*/
template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistanceLine2Line(const VEC3& A, const VEC3& AB, typename VEC3::DATA_TYPE AB2, const VEC3& P, const VEC3& Q) ;

/**
* compute squared distance from line to segment
* @return the squared distance
*/
template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistanceLine2Seg(const VEC3& A, const VEC3& AB, typename VEC3::DATA_TYPE AB2, const VEC3& P, const VEC3& Q) ;

/**
* compute squared distance from segment to point
* @return the squared distance
*/
template <typename VEC3>
typename VEC3::DATA_TYPE squaredDistanceSeg2Point(const VEC3& A, const VEC3& AB, typename VEC3::DATA_TYPE AB2, const VEC3& P) ;


/**
 * compute the closest points on a line from an other
 * @param P1 one point on first line
 * @param V1 vector defining first line
 * @param P2 one point on second line
 * @param P1 vector defining second line
 * @param Q1 computed point of (P1,V1) closest of (P2,V2)
 * @param Q2 computed point of (P2,V2) closest of (P1,V1)
 * &return true if computaable
 */
template <typename VEC3>
bool lineLineClosestPoints(const VEC3& P1, const VEC3& V1, const VEC3& P2, const VEC3& V2, VEC3& Q1, VEC3& Q2);

} // namespace Geom

} // namespace CGoGN

#include "Geometry/distances.hpp"

#endif
