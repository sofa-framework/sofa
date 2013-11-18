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

template <unsigned int DIM, typename T>
std::string Vector<DIM, T>::CGoGNnameOfType()
{
	std::stringstream ss ;
	ss << "Geom::Vector<" ;
	ss << DIM ;
	ss << "," ;
	ss << nameOfType(T()) ;
	ss << ">" ;

	return ss.str() ;
}

/**********************************************/
/*                CONSTRUCTORS                */
/**********************************************/

template <unsigned int DIM, typename T>
Vector<DIM, T>::Vector()
{
	CGoGN_STATIC_ASSERT(DIM > 0, invalid_zero_dimensional_Vector) ;
	zero() ;
}

template <unsigned int DIM, typename T>
Vector<DIM, T>::Vector(const Vector<DIM, T>& v)
{
	for (unsigned int i = 0; i < DIM; ++i)
		m_data[i] = v[i] ;
}

template <unsigned int DIM, typename T>
template <typename T2>
Vector<DIM, T>::Vector(const Vector<DIM, T2>& v)
{
	for (unsigned int i = 0; i < DIM; ++i)
		m_data[i] = T(v[i]) ;
}

template <unsigned int DIM, typename T>
Vector<DIM, T>::Vector(T x, T y)
{
	CGoGN_STATIC_ASSERT(DIM == 2, incompatible_Vector_constructor_dimension) ;
	m_data[0] = x ;
	m_data[1] = y ;
}

template <unsigned int DIM, typename T>
Vector<DIM, T>::Vector(T x, T y, T z)
{
	CGoGN_STATIC_ASSERT(DIM == 3, incompatible_Vector_constructor_dimension) ;
	m_data[0] = x ;
	m_data[1] = y ;
	m_data[2] = z ;
}

template <unsigned int DIM, typename T>
Vector<DIM, T>::Vector(T x, T y, T z, T w)
{
	CGoGN_STATIC_ASSERT(DIM == 4, incompatible_Vector_constructor_dimension) ;
	m_data[0] = x ;
	m_data[1] = y ;
	m_data[2] = z ;
	m_data[3] = w ;
}

template <unsigned int DIM, typename T>
Vector<DIM, T>::Vector(T x)
{
	set(x) ;
}

template <unsigned int DIM, typename T>
inline void Vector<DIM, T>::set(T a)
{
	for (unsigned int i = 0; i < DIM; ++i)
		m_data[i] = a ;
}

template <unsigned int DIM, typename T>
inline void Vector<DIM, T>::zero()
{
	set(T(0)) ;
}

/**********************************************/
/*                 ACCESSORS                  */
/**********************************************/

template <unsigned int DIM, typename T>
inline T& Vector<DIM, T>::operator[](unsigned int index)
{
	assert(index < DIM) ;
	return m_data[index] ;
}

template <unsigned int DIM, typename T>
inline const T& Vector<DIM, T>::operator[](unsigned int index) const
{
	assert(index < DIM) ;
	return m_data[index] ;
}

template <unsigned int DIM, typename T>
inline unsigned int Vector<DIM, T>::dimension() const
{
	return DIM ;
}

template <unsigned int DIM, typename T>
inline T* Vector<DIM, T>::data()
{
	return m_data ;
}

template <unsigned int DIM, typename T>
inline const T* Vector<DIM, T>::data() const
{
	return m_data ;
}

/**********************************************/
/*         ARITHMETIC SELF-OPERATORS          */
/**********************************************/

template <unsigned int DIM, typename T>
inline Vector<DIM, T>& Vector<DIM, T>::operator+=(const Vector<DIM, T>& v)
{
	for (unsigned int i = 0; i < DIM; ++i)
		m_data[i] += v[i] ;
	return *this ;
}

template <unsigned int DIM, typename T>
inline Vector<DIM, T>& Vector<DIM, T>::operator-=(const Vector<DIM, T>& v)
{
	for (unsigned int i = 0; i < DIM; ++i)
		m_data[i] -= v[i] ;
	return *this ;
}

template <unsigned int DIM, typename T>
inline Vector<DIM, T>& Vector<DIM, T>::operator*=(T a)
{
	for (unsigned int i = 0; i < DIM; ++i)
		m_data[i] *= a ;
	return *this ;
}

template <unsigned int DIM, typename T>
inline Vector<DIM, T>& Vector<DIM, T>::operator/=(T a)
{
	for (unsigned int i = 0; i < DIM; ++i)
		m_data[i] /= a ;
	return *this ;
}

/**********************************************/
/*            ARITHMETIC OPERATORS            */
/**********************************************/

template <unsigned int DIM, typename T>
inline Vector<DIM, T> Vector<DIM, T>::operator+(const Vector<DIM, T>& v) const
{
	Vector<DIM, T> res ;
	for (unsigned int i = 0; i < DIM; ++i)
		res[i] = m_data[i] + v[i] ;
	return res ;
}

template <unsigned int DIM, typename T>
inline Vector<DIM, T> Vector<DIM, T>::operator-(const Vector<DIM, T>& v) const
{
	Vector<DIM, T> res ;
	for (unsigned int i = 0; i < DIM; ++i)
		res[i] = m_data[i] - v[i] ;
	return res ;
}

template <unsigned int DIM, typename T>
inline Vector<DIM, T> Vector<DIM, T>::operator-() const
{
	Vector<DIM, T> res ;
	for (unsigned int i = 0; i < DIM; ++i)
		res[i] = - m_data[i] ;
	return res ;
}


template <unsigned int DIM, typename T>
inline Vector<DIM, T> Vector<DIM, T>::operator*(T a) const
{
	Vector<DIM, T> res ;
	for (unsigned int i = 0; i < DIM; ++i)
		res[i] = m_data[i] * a ;
	return res ;
}

template <unsigned int DIM, typename T>
inline Vector<DIM, T> Vector<DIM, T>::operator/(T a) const
{
	Vector<DIM, T> res ;
	for (unsigned int i = 0; i < DIM; ++i)
		res[i] = m_data[i] / a ;
	return res ;
}

/**********************************************/
/*             UTILITY FUNCTIONS              */
/**********************************************/

template <unsigned int DIM, typename T>
inline T Vector<DIM, T>::norm2() const
{
	T n(0) ;
	for (unsigned int i = 0; i < DIM; ++i)
		n += m_data[i] * m_data[i] ;
	return n ;
}

template <unsigned int DIM, typename T>
inline double Vector<DIM, T>::norm() const
{
	return sqrt(norm2()) ;
}

template <unsigned int DIM, typename T>
inline double Vector<DIM, T>::normalize()
{
	double n = norm() ;
	if (n != T(0)) for (unsigned int i = 0; i < DIM; ++i)
		m_data[i] /= T(n) ;
	return n ;
}

template <unsigned int DIM, typename T>
inline Vector<DIM, T> Vector<DIM, T>::normalized() const
{
	Vector<DIM, T> v(*this);
	v.normalize();
	return v;
}

template <unsigned int DIM, typename T>
inline T Vector<DIM, T>::operator*(const Vector<DIM, T> v) const
{
	T d(0) ;
	for (unsigned int i = 0; i < DIM; ++i)
		d += m_data[i] * v[i] ;
	return d ;
}

template <unsigned int DIM, typename T>
inline Vector<DIM, T> Vector<DIM, T>::operator^(const Vector<DIM, T> v) const
{
	CGoGN_STATIC_ASSERT(DIM == 3, incompatible_Vector_cross_product_dimension) ;
	Vector<DIM, T> c ;
	c[0] = m_data[1] * v[2] - m_data[2] * v[1] ;
	c[1] = m_data[2] * v[0] - m_data[0] * v[2] ;
	c[2] = m_data[0] * v[1] - m_data[1] * v[0] ;
	return c ;
}

template <unsigned int DIM, typename T>
inline bool Vector<DIM, T>::operator==(const Vector<DIM, T>& v) const
{
	for (unsigned int i = 0; i < DIM; ++i)
		if (v[i] != m_data[i]) return false ;
	return true ;
}

template <unsigned int DIM, typename T>
inline bool Vector<DIM, T>::operator!=(const Vector<DIM, T>& v) const
{
	for (unsigned int i = 0; i < DIM; ++i)
		if (v[i] != m_data[i]) return true ;
	return false ;
}

template <unsigned int DIM, typename T>
inline bool Vector<DIM, T>::hasNan() const
{
	for (unsigned int i = 0; i < DIM; ++i)
		if (m_data[i] != m_data[i]) return true ;
	return false ;
}

template <unsigned int DIM, typename T>
inline bool Vector<DIM, T>::isFinite() const
{
	for (unsigned int i = 0; i < DIM; ++i)
		if (!std::isfinite(m_data[i])) return false ;
	return true ;
}

template <unsigned int DIM, typename T>
inline bool Vector<DIM, T>::isNormalized(const T& epsilon) const
{
	return (1 - epsilon < norm2() && norm2() < 1 + epsilon) ;
}

template <unsigned int DIM, typename T>
inline bool Vector<DIM, T>::isOrthogonal(const Vector<DIM, T>& v, const T& epsilon) const
{
	return (fabs(v * (*this)) < epsilon) ;
}

template <unsigned int DIM, typename T>
inline bool Vector<DIM, T>::isNear(const Vector<DIM, T>& v, int precision) const
{
	T diff ;
	T norm2(0) ;
	for (unsigned int i = 0 ; i < DIM ; ++i)
	{
		diff = m_data[i] - v[i] ;
		if (!isNull(diff, precision)) return false ;
		norm2 += diff * diff ;
	}
	return isNull2(norm2, precision) ;
}

/**********************************************/
/*             STREAM OPERATORS               */
/**********************************************/

template <unsigned int DIM, typename T>
std::ostream& operator<<(std::ostream& out, const Vector<DIM, T>& v)
{
	for (unsigned int i = 0; i < DIM; ++i)
		out << v[i] << " " ;
	return out ;
}

template <unsigned int DIM, typename T>
std::istream& operator>>(std::istream& in, Vector<DIM, T>& v)
{
	for (unsigned int i = 0; i < DIM; ++i)
		in >> v[i] ;
	return in ;
}

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
inline Vector<DIM, T> operator*(T2 b, const Vector<DIM, T>& v)
{
	return v * T(b) ;
}


template <unsigned int DIM, typename T>
inline Vector<DIM, T> operator/(T a, const Vector<DIM, T>& v)
{
	return v / a ;
}

template <unsigned int DIM, typename T>
inline T tripleProduct(const Vector<DIM, T>& v1, const Vector<DIM, T>& v2, const Vector<DIM, T>& v3)
{
	return v1 * (v2 ^ v3) ;
}

template <unsigned int DIM, typename T>
inline Vector<DIM, T> slerp(const Vector<DIM, T>& v1, const Vector<DIM, T>& v2, const T& t)
{
	Vector<DIM, T> res ;

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
