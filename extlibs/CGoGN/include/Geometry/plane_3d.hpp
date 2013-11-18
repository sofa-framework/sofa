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

namespace CGoGN
{

namespace Geom
{

template <typename T>
std::string Plane3D<T>::CGoGNnameOfType()
{
	std::stringstream ss ;
	ss << "Geom::Plane3D<" ;
	ss << nameOfType(T()) ;
	ss << ">" ;

	return ss.str() ;
}

/**********************************************/
/*                CONSTRUCTORS                */
/**********************************************/

template <typename T>
Plane3D<T>::Plane3D(int d) : m_normal(0), m_d(d)
{ }

template <typename T>
Plane3D<T>::Plane3D(const Plane3D<T>& p)
{
	m_normal = p.m_normal ;
	m_d = p.m_d ;
}

template <typename T>
Plane3D<T>::Plane3D(const Vector<3,T>& n, T d) : m_normal(n), m_d(d)
{
	m_normal.normalize();
}

template <typename T>
Plane3D<T>::Plane3D(const Vector<3,T>& n, const Vector<3,T>& p) : m_normal(n), m_d(-(p*n))
{
	m_normal.normalize();
}

template <typename T>
Plane3D<T>::Plane3D(const Vector<3,T>& p1, const Vector<3,T>& p2, const Vector<3,T>& p3)
{
	m_normal = (p2-p1) ^ (p3-p1) ;
	m_normal.normalize() ;
	m_d = -(p1 * m_normal) ;
}

/**********************************************/
/*                 ACCESSORS                  */
/**********************************************/
template <typename T>
Vector<3,T>& Plane3D<T>::normal()
{
	return m_normal ;
}

template <typename T>
const Vector<3,T>& Plane3D<T>::normal() const
{
	return m_normal ;
}

template <typename T>
T& Plane3D<T>::d()
{
	return m_d ;
}

template <typename T>
const T& Plane3D<T>::d() const
{
	return m_d ;
}

/**********************************************/
/*             UTILITY FUNCTIONS              */
/**********************************************/

template <typename T>
T Plane3D<T>::distance(const Vector<3,T>& p) const
{
	T k = m_normal * p ;
	return k + m_d ;
}

template <typename T>
void Plane3D<T>::project(Vector<3,T>& p) const
{
#define PRECISION 1e-10
	T d = -distance(p) ;
	if(fabs(d) > PRECISION)
	{
		Vector<3,T> v = m_normal * d ;
		p += v ;
	}
#undef PRECISION
}

template <typename T>
Orientation3D Plane3D<T>::orient(const Vector<3,T>& p) const
{
#define PRECISION 1e-6
	T dist = distance(p) ;

	if(dist < -PRECISION)
		return UNDER ;
	if(dist > PRECISION)
		return OVER ;
	return ON ;
#undef PRECISION
}

/**********************************************/
/*             STREAM OPERATORS               */
/**********************************************/

template <typename T>
std::ostream& operator<<(std::ostream& out, const Plane3D<T>& p)
{
	out << p.normal() << " " << p.d() ;
	return out ;
}

template <typename T>
std::istream& operator>>(std::istream& in, Plane3D<T>& p)
{
	in >> p.normal() >> p.d() ;
	return in ;
}

} // namespace Geom

} // namespace CGoGN
