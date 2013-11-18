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

#include <Eigen/Dense>
#include "Utils/convertType.h"

namespace CGoGN
{

namespace Geom
{

template <typename VEC3>
Inclusion isPointInTriangle(const VEC3& point,const VEC3& Ta,const VEC3& Tb,const VEC3& Tc)
{
	typedef typename VEC3::DATA_TYPE T ;

	//using barycentric coordinate
	VEC3 v0(Tc) ;
	v0 -= Ta ;
	VEC3 v1(Tb) ;
	v1 -= Ta ;
	VEC3 v2(point) ;
	v2 -= Ta ;

	// Compute dot products
	T dot00 = v0 * v0 ;
	T dot01 = v0 * v1 ;
	T dot02 = v0 * v2 ;
	T dot11 = v1 * v1 ;
	T dot12 = v1 * v2 ;

	// Compute barycentric coordinates
	T invDenom = T(1) / (dot00 * dot11 - dot01 * dot01) ;
	T u = (dot11 * dot02 - dot01 * dot12) * invDenom ;
	T v = (dot00 * dot12 - dot01 * dot02) * invDenom ;

	// Check if point is in triangle
	if(u == T(0))
	{
		if(v == T(0))
			return VERTEX_INCLUSION ;
		if(v < T(1) && v > T(0))
			return EDGE_INCLUSION ;
		if(v == T(1))
			return VERTEX_INCLUSION ;
	}
	else if(u > T(0) && u < T(1))
	{
		if(v == T(0))
			return EDGE_INCLUSION ;
		if(v > T(0))
		{
			if(u+v < T(1))
				return FACE_INCLUSION ;
			else if(u+v == T(1))
				return EDGE_INCLUSION ;
		}
	}
	else if(u == T(1) && v == T(0))
		return VERTEX_INCLUSION ;

	return NO_INCLUSION ;
}

template <typename VEC3>
bool isPointInSphere(const VEC3& point, const VEC3& center, const typename VEC3::DATA_TYPE& radius)
{
	return (point - center).norm() < radius;
}

template <typename VEC3>
Inclusion isSegmentInTriangle2D(const VEC3& P1, const VEC3& P2, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc, const VEC3& N)
{
	Orientation2D oP1P2A = testOrientation2D(Ta,P1,P2,N) ;
	Orientation2D oP1P2B = testOrientation2D(Tb,P1,P2,N) ;
	Orientation2D oP1P2C = testOrientation2D(Tc,P1,P2,N) ;

	if(oP1P2A == oP1P2B && oP1P2B == oP1P2C)
		return NO_INCLUSION ;

	Orientation2D oABP1 = testOrientation2D(P1,Ta,Tb,N) ;
	Orientation2D oABP2 = testOrientation2D(P2,Ta,Tb,N) ;

	Orientation2D oBCP1 = testOrientation2D(P1,Tb,Tc,N) ;
	Orientation2D oBCP2 = testOrientation2D(P2,Tb,Tc,N) ;

	Orientation2D oCAP1 = testOrientation2D(P1,Tc,Ta,N) ;
	Orientation2D oCAP2 = testOrientation2D(P2,Tc,Ta,N) ;

	unsigned int diff=0 ;
	if(oABP1!=oABP2)
		++diff ;
	if(oBCP1!=oBCP2)
		++diff ;
	if(oCAP1!=oCAP2)
		++diff ;

	if(diff < 2)
		return NO_INCLUSION ;

	return FACE_INCLUSION ;
}

template <typename VEC3>
bool isPointInTetrahedron(VEC3 points[4], VEC3& point, bool CCW)
{
	typedef typename VEC3::DATA_TYPE T ;

	VEC3 AB = points[1] - points[0] ;
	VEC3 AC = points[2] - points[0] ;
	VEC3 AD = points[3] - points[0] ;

	Eigen::Matrix3f A;
	A << AB[0], AB[1], AB[2],
			AC[0], AC[1], AC[2],
			AD[0], AD[1], AD[2];

	Eigen::Matrix3f AInv = A.inverse();

	VEC3 v1(point-points[0]);
	Eigen::Vector3f& v = Utils::convertRef<Eigen::Vector3f>(v1);
	Eigen::Vector3f beta = AInv* v;
	return (beta[0] >= 0.0f && beta[1] >= 0.0f && beta[2] >= 0.0f && beta[0]+beta[1]+beta[2]<=1.0f);
}

template <typename VEC3>
bool isEdgeInOrIntersectingTetrahedron(VEC3 points[4], VEC3& point1, VEC3& point2, bool CCW)
{
	if(isPointInTetrahedron(points,point1,CCW) || isPointInTetrahedron(points,point1,CCW))
		return true ;

	VEC3 inter;
	if(intersectionSegmentTriangle(point1, point2, points[0], points[1], points[2], inter)
	|| intersectionSegmentTriangle(point1, point2, points[0], points[1], points[3], inter)
	|| intersectionSegmentTriangle(point1, point2, points[1], points[2], points[3], inter)
	|| intersectionSegmentTriangle(point1, point2, points[2], points[0], points[3], inter))
		return true ;

	return false ;
}

template <typename VEC>
bool arePointsEquals(const VEC& point1, const VEC& point2)
{
	VEC v(point1 - point2);

#define PRECISION 1e-5
	return v.norm2() <= PRECISION ;
#undef PRECISION
}

} // namespace Geom

} // namespace CGoGN
