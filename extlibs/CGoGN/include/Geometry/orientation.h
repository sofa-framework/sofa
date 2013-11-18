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

#ifndef __ORIENTATION__
#define __ORIENTATION__

#include "Geometry/basic.h"

namespace CGoGN
{

namespace Geom
{

enum OrientationLine
{
	CW, CCW, INTERSECT
} ;

enum Orientation2D
{
	ALIGNED, RIGHT, LEFT
} ;

/**
 * return the relative orientation of oriented lines (a,b) and (c,d)
 * when looking in the direction of (a,b) tells if (c,d) turns CW or CCW around (a,b)
 * return INTERSECT if (a,b) and (c,d) intersect each other
 */
template <typename VEC3>
OrientationLine testOrientationLines(const VEC3& a, const VEC3& b, const VEC3& c, const VEC3& d)
{
	typedef typename VEC3::DATA_TYPE T ;
	T vol = tetraSignedVolume(a, b, c, d) ;
	return vol > T(0) ? CCW : vol < T(0) ? CW : INTERSECT ;
}

/**
 * return the orientation of point P w.r.t. the plane defined by 3 points
 * @param P the point
 * @param A plane point 1
 * @param B plane point 2
 * @param C plane point 3
 * @return the orientation
 */
template <typename VEC3>
Orientation3D testOrientation3D(const VEC3& P, const VEC3& A, const VEC3& B, const VEC3& C)
{
	typedef typename VEC3::DATA_TYPE T ;
	Geom::Plane3D<T> plane(A, B, C) ;
	return plane.orient(P) ;
}

/**
 * return the orientation of point P w.r.t. the plane defined by its normal and 1 point
 * @param P the point
 * @param N plane normal
 * @param PP plane point
 * @return the orientation
 */
template <typename VEC3>
Orientation3D testOrientation3D(const VEC3& P, const VEC3& N, const VEC3& PP)
{
	typedef typename VEC3::DATA_TYPE T ;
	Geom::Plane3D<T> plane(N, PP) ;
	return plane.orient(P) ;
}

/**
 * return the orientation of point P w.r.t. the vector (Pb-Pa)
 * --> tells if P is on/right/left of the line (Pa,Pb)
 * @param P the point
 * @param Pa origin point
 * @param Pb end point
 * @return the orientation
 */
template <typename VEC3>
Orientation2D testOrientation2D(const VEC3& P, const VEC3& Pa, const VEC3& Pb) ;

/**
 * return the relative orientation of two vectors in the plane (u,v)
 * the return value is
 *  +1 if u^v > 0
 *   0 if u^v = 0
 *  -1 if u^v < 0
 * @param u first vector
 * @param v second vector
 * @return the orientation
 */
template <typename VEC3>
int orientation2D(const VEC3& u, const VEC3& v) ;

/**
 * test if two vectors are aligned or orthogonal, the return value is
 *  +1 if u and v are ALIGNED and u*v > 0
 *   0 if u and v are ORTHOGONAL or u*v = 0
 *  -1 if u and v are ALIGNED and u*v < 0
 * @param u first vector
 * @param v second vector
 * @return the alignment
 */
template <typename VEC3>
int aligned2D(const VEC3& u, const VEC3& v) ;

/**
 * test if vector u is between vectors v and w in the plane (v,w)
 * in other words if u is inside the angular sector [v,w[
 * (v being included and w being is excluded)
 * if u,v,w are aligned the result is true
 *
 * @param u first vector of the angular sector
 * @param v second vector of the angular sector
 * @param w the vector to test
 * @return the result of the test
 */
template <typename VEC3>
bool isBetween(const VEC3& u, const VEC3& v, const VEC3& w) ;

/**
 * test if the tetrahedron is well oriented depending on the orientation of the faces we want
 * @param the list of the points of the tetra (0,1,2) the first face in the orientation we want and (3) the last point of the tetra
 * @param true if the faces of the tetra must be in CCW order (default=true)
 */
template <typename VEC3>
bool isTetrahedronWellOriented(const VEC3 points[4], bool CCW = true) ;

}

}

#include "orientation.hpp"

#endif
