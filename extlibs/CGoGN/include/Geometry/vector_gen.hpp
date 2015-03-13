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

#include <sstream>
#include <cmath>

namespace CGoGN
{

namespace Geom
{



/***
 * Test if x is null within precision.
 * 3 cases are possible :
 *  - precision = 0 : x is null <=> (x == 0)
 *  - precision > 0 : x is null <=> (|x| < precision)
 *  - precision < 0 : x is null <=> (|x| < 1/precision) <=> (precision * |x| < 1)
 */
template <typename T>
inline bool isNull(T x, int precision)
{
	if (precision == 0)
		return (x == 0) ;
	else if (precision > 0)
			return (fabs(x) < precision) ;
	else
		return (precision * fabs(x) < 1) ;
}

/***
 * Test if the square root of x is null within precision.
 * In other words, test if x is null within precision*precision
 */
template <typename T>
inline bool isNull2(T x, int precision)
{
	if (precision == 0)
		return (x == 0) ;
	else if (precision > 0)
		return (isNull(x, precision * precision)) ;
	else
		return (isNull(x, - (precision * precision))) ;
}

template <unsigned int DIM, typename T, typename T2>
inline typename Vector<DIM, T>::type operator*(T2 b, const typename Vector<DIM, T>::type& v)
{
	return v * T(b) ;
}


template <unsigned int DIM, typename T>
inline typename Vector<DIM, T>::type operator/(T a, const typename Vector<DIM, T>::type& v)
{
	return v / a ;
}

template <unsigned int DIM, typename T>
inline T tripleProduct(const sofa::defaulttype::Vec<DIM, T>& v1, const  sofa::defaulttype::Vec<DIM, T>& v2, const  sofa::defaulttype::Vec<DIM, T>& v3)
{
    return static_cast<T>(v1 * (v2.cross(v3))) ;
}

template <unsigned int DIM, typename T>
inline typename Vector<DIM, T>::type slerp(const typename Vector<DIM, T>::type& v1, const typename Vector<DIM, T>::type& v2, const T& t)
{
    typename Vector<DIM, T>::type res ;

	T scal = v1 * v2 ;

	// Prevention for floating point errors
	if (1 < scal && scal < 1 + 1e-6)
		scal = T(1) ;
	if (-1. - 1e-6 < scal && scal < -1)
		scal = -T(1) ;

	T omega = acos(scal) ;
	T den = sin(omega) ;

	if (-1e-8 < den && den < 1e-8) return t < 0.5 ? v1 : v2 ;

	T f1 = sin((T(1) - t) * omega) / den ;	// assume 0 <= t <= 1
	T f2 = sin(t * omega) / den ;

	res += f1 * v1 ;
	res += f2 * v2 ;

	return res ;
}

} // namespace Geom

} // namespace CGoGN
