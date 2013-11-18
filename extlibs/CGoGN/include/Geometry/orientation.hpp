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

#include <limits>

namespace CGoGN
{

namespace Geom
{

template <typename VEC3>
Orientation2D testOrientation2D(const VEC3& P, const VEC3& Pa, const VEC3& Pb)
{

	typedef typename VEC3::DATA_TYPE T ;
	const T zero = 0.000001 ;

	T p = (P[0] - Pa[0]) * (Pb[1] - Pa[1]) - (Pb[0] - Pa[0]) * (P[1] - Pa[1]) ;

	if (p > zero)
		return RIGHT ;
	else if (-p > zero)
		return LEFT ;
	else
		return ALIGNED ;
}

// TODO use triple product with a normal to the plane that contains u and v
template <typename VEC3>
int orientation2D(const VEC3& u, const VEC3& v)
{
	typedef typename VEC3::DATA_TYPE T ;

	T p = u[0] * v[1] - u[1] * v[0] ;
	const T zero = 0.0001 ;

	if (p > zero)
		return 1 ;
	else if (-p > zero)
		return -1 ;
	else
		return 0 ;
}

// TODO use dot product => include epsilon in vector_gen to test sign
template <typename VEC3>
int aligned2D(const VEC3& u, const VEC3& v)
{
	typedef typename VEC3::DATA_TYPE T ;

	T p = u[0] * v[0] + u[1] * v[1] ;
	const T zero = 0.0001 ;

	if (p > zero)
		return 1 ;
	else if (-p > zero)
		return -1 ;
	else
		return 0 ;
}

template <typename VEC3>
bool isBetween(const VEC3& u, const VEC3& v, const VEC3& w)
{
	int orientWV = orientation2D(w,v) ;

	if (orientWV > 0)
	{
		if (orientation2D(v,u) >= 0) return true ;
		int orientWU = orientation2D(w,u) ;
		if (orientWU < 0) return true ;
		return (orientWU == 0) && (aligned2D(w,u) <= 0) ;
	}
	else if (orientWV < 0 || (orientWV == 0 && aligned2D(w,v) < 0))
	{
		if (orientation2D(v,u) < 0) return false ;
		int orientWU = orientation2D(w,u) ;
		if (orientWU < 0) return true ;
		return (orientWU == 0) && (aligned2D(w,u) <= 0) ;
	}
	else // orientWV == 0 && v*u >= 0
	     // ==> v et u ont même direction ou sont nuls
	{
		return (orientation2D(v,u) == 0 && aligned2D(v,u) >= 0) ;
	}
}

template <typename VEC3>
bool isTetrahedronWellOriented(const VEC3 points[4], bool CCW)
{
	typedef typename VEC3::DATA_TYPE T ;

	VEC3 AB = points[1] - points[0] ;
	VEC3 AC = points[2] - points[0] ;
	VEC3 AD = points[3] - points[0] ;

	VEC3 N = AB ^ AC ;

	T dot = N * AD ;
	if (CCW)
		return dot <= 0 ;
	else
		return dot >= 0 ;
}

}

}
