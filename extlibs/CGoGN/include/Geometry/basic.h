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

#ifndef __GEOMETRY__
#define __GEOMETRY__

#include "Geometry/vector_gen.h"
#include "Geometry/plane_3d.h"

namespace CGoGN
{

namespace Geom
{

// linear interpolation between 2 points
template <typename VEC>
VEC lerp(const VEC& v1, const VEC& v2, typename VEC::DATA_TYPE s)
{
	return (1.0 - s) * v1 + s * v2 ;
}

// weighted barycenter of 2 points
template <unsigned int DIM, typename T>
typename Vector<DIM,T>::type barycenter(const typename Vector<DIM,T>::type& v1, const typename Vector<DIM,T>::type& v2, T a, T b)
{
	return a * v1 + b * v2 ;
}

// isobarycenter of 2 points
template <unsigned int DIM, typename T>
typename Vector<DIM,T>::type isobarycenter(const typename Vector<DIM,T>::type& v1, const typename Vector<DIM,T>::type& v2)
{
	return lerp(v1, v2, 0.5) ;
}

// weighted barycenter of 3 points
template <unsigned int DIM, typename T>
typename Vector<DIM,T>::type barycenter(const typename Vector<DIM,T>::type& v1, const typename Vector<DIM,T>::type& v2, const typename Vector<DIM,T>::type& v3, T a, T b, T c)
{
	return a * v1 + b * v2 + c * v3 ;
}

// isobarycenter of 3 points
template <unsigned int DIM, typename T>
typename Vector<DIM,T>::type isobarycenter(const typename Vector<DIM,T>::type& v1, const typename Vector<DIM,T>::type& v2, const typename Vector<DIM,T>::type& v3)
{
        typename Vector<DIM,T>::type v ;
	for(unsigned int i = 0; i < DIM; ++i)
		v[i] = (v1[i] + v2[i] + v3[i]) / T(3) ;
	return v ;
}

// cosinus of the angle formed by 2 vectors
template <typename VEC>
typename VEC::DATA_TYPE cos_angle(const VEC& a, const VEC& b)
{
	typename VEC::DATA_TYPE na2 = a.norm2() ;
	typename VEC::DATA_TYPE nb2 = b.norm2() ;
	return (a * b) / sqrt(na2 * nb2) ;
}

// angle formed by 2 vectors
template <typename VEC>
typename VEC::DATA_TYPE angle(const VEC& a, const VEC& b)
{
	return acos(cos_angle(a,b)) ;
}

// area of the triangle formed by 3 points in 3D
template <typename VEC3>
typename VEC3::value_type triangleArea(const VEC3& p1, const VEC3& p2, const VEC3& p3)
{
	return 0.5 * ((p2 - p1) ^ (p3 - p1)).norm() ;
}

// normal of the plane spanned by 3 points in 3D
template <typename VEC3>
VEC3 triangleNormal(const VEC3& p1, const VEC3& p2, const VEC3& p3)
{
	return (p2 - p1) ^ (p3 - p1) ;
}

// return true if the triangle formed by 3 points in 3D is obtuse, false otherwise
template <typename VEC3>
bool isTriangleObtuse(const VEC3& p1, const VEC3& p2, const VEC3& p3)
{
    typename VEC3::value_type a1 = angle(p2 - p1, p3 - p1) ;
	if(a1 > M_PI / 2)
		return true ;
    typename VEC3::value_type a2 = angle(p3 - p2, p1 - p2) ;
	if(a2 > M_PI / 2 || a1 + a2 < M_PI / 2)
		return true ;
	return false ;
}

// signed volume of the tetrahedron formed by 4 points in 3D
template <typename VEC3>
typename VEC3::value_type tetraSignedVolume(const VEC3& p1, const VEC3& p2, const VEC3& p3, const VEC3& p4)
{
    typedef typename VEC3::value_type T;
    return tripleProduct<3,T>(p2 - p1, p3 - p1, p4 - p1) / T(6) ;
}

// volume of the tetrahedron formed by 4 points in 3D
template <typename VEC3>
typename VEC3::value_type tetraVolume(const VEC3& p1, const VEC3& p2, const VEC3& p3, const VEC3& p4)
{
	return fabs(tetraSignedVolume(p1,p2,p3,p4)) ;
}

// volume of the parallelepiped spanned by three 3D vectors
template <typename VEC3>
typename VEC3::value_type parallelepipedVolume(const VEC3& v1, const VEC3& v2, const VEC3& v3)
{
	return tripleProduct(v1, v2, v3) ;
}

} // namespace Geom

} // namespace CGoGN

#endif
