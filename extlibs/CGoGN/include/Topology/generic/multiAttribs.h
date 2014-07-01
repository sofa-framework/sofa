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

#ifndef __MULTI2ATTRIBS_H_
#define __MULTI2ATTRIBS_H_

#include "Topology/generic/cells.h"

namespace CGoGN
{

//forward
template <typename T1, typename T2>
struct RefCompo2Type;


template <typename T1, typename T2>
struct Compo2Type
{
	T1 m_v1;
	T2 m_v2;

	Compo2Type(){}
	Compo2Type(double z ): m_v1(z), m_v2(z) {}

	Compo2Type(const Compo2Type<double,double>& ct ): m_v1(ct.m_v1), m_v2(ct.m_v1) {}

	Compo2Type(const RefCompo2Type<T1,T2>& v);

	Compo2Type<T1,T2>& operator =(const Compo2Type<T1,T2>& comp);

	Compo2Type<T1,T2> operator+(const Compo2Type<T1,T2>& v) const;
	Compo2Type<T1,T2> operator-(const Compo2Type<T1,T2>& v) const;
	Compo2Type<T1,T2> operator/(double d) const;
	Compo2Type<T1,T2> operator*(double d) const;
	Compo2Type<T1,T2>& operator+=(const Compo2Type<T1,T2>& v);
	Compo2Type<T1,T2>& operator-=(const Compo2Type<T1,T2>& v);
	Compo2Type<T1,T2>& operator*=(double d);
	Compo2Type<T1,T2>& operator/=(double d);
};


template <typename T1, typename T2>
struct RefCompo2Type
{
	T1& m_v1;
	T2& m_v2;

	RefCompo2Type(T1& v1, T2& v2): m_v1(v1), m_v2(v2) {}
	RefCompo2Type (Compo2Type<T1,T2>& comp);

	RefCompo2Type<T1,T2>& operator=(const RefCompo2Type<T1,T2>& v);
	RefCompo2Type<T1,T2>& operator=(Compo2Type<T1,T2> v);

	Compo2Type<T1,T2> operator+(const RefCompo2Type<T1,T2>& v) const;
	Compo2Type<T1,T2> operator-(const RefCompo2Type<T1,T2>& v) const;
	Compo2Type<T1,T2> operator/(double d) const;
	Compo2Type<T1,T2> operator*(double d) const;
	RefCompo2Type<T1,T2>& operator+=(const RefCompo2Type<T1,T2>& v);
	RefCompo2Type<T1,T2>& operator-=(const RefCompo2Type<T1,T2>& v);
	RefCompo2Type<T1,T2>& operator*=(double d);
	RefCompo2Type<T1,T2>& operator/=(double d);
};

template <typename T1, typename T2>
Compo2Type<double, double> length( const Compo2Type<T1,T2>& v)
{
	Compo2Type<double, double> l;
	l.m_v1 = sqrt(v.m_v1*v.m_v1);
	l.m_v2 = sqrt(v.m_v2*v.m_v2);
	return l;
}

template <typename T>
double length(const T& v)
{
	return v.norm();
}


template <typename T1, typename T2, typename MAP>
class Vertex2Attributes
{
	VertexAttribute<T1, MAP>& m_h1;
	VertexAttribute<T2, MAP>& m_h2;
public:
	typedef Compo2Type<T1,T2> DATA_TYPE;
	typedef RefCompo2Type<T1,T2> REF_DATA_TYPE;

	Vertex2Attributes(VertexAttribute<T1, MAP>& h1, VertexAttribute<T2, MAP>& h2):
		m_h1(h1), m_h2(h2) {}

	RefCompo2Type<T1,T2> operator[](unsigned int a)
	{
		return RefCompo2Type<T1,T2>(m_h1[a],m_h2[a]);
	}

	RefCompo2Type<T1,T2> operator[](Vertex d)
	{
		return RefCompo2Type<T1,T2>(m_h1[d],m_h2[d]);
	}

	const RefCompo2Type<T1,T2> operator[](unsigned int a) const
	{
		return RefCompo2Type<T1,T2>(m_h1[a],m_h2[a]);
	}

	const RefCompo2Type<T1,T2> operator[](Vertex d) const
	{
		return RefCompo2Type<T1,T2>(m_h1[d],m_h2[d]);
	}

	static unsigned int getOrbit() { return VERTEX; }
};

template <typename T1, typename T2, typename MAP>
class Edge2Attributes
{
	EdgeAttribute<T1, MAP>& m_h1;
	EdgeAttribute<T2, MAP>& m_h2;
public:
	typedef Compo2Type<T1,T2> DATA_TYPE;
	typedef RefCompo2Type<T1,T2> REF_DATA_TYPE;

	Edge2Attributes(EdgeAttribute<T1, MAP>& h1, EdgeAttribute<T2, MAP>& h2):
		m_h1(h1), m_h2(h2) {}

	RefCompo2Type<T1,T2> operator[](unsigned int a)
	{
		return RefCompo2Type<T1,T2>(m_h1[a],m_h2[a]);
	}

	RefCompo2Type<T1,T2> operator[](Edge d)
	{
		return RefCompo2Type<T1,T2>(m_h1[d],m_h2[d]);
	}

	const RefCompo2Type<T1,T2> operator[](unsigned int a) const
	{
		return RefCompo2Type<T1,T2>(m_h1[a],m_h2[a]);
	}

	const RefCompo2Type<T1,T2> operator[](Edge d) const
	{
		return RefCompo2Type<T1,T2>(m_h1[d],m_h2[d]);
	}

	static unsigned int getOrbit() { return EDGE; }
};

template <typename T1, typename T2, typename MAP>
class Face2Attributes
{
	FaceAttribute<T1, MAP>& m_h1;
	FaceAttribute<T2, MAP>& m_h2;
public:
	typedef Compo2Type<T1,T2> DATA_TYPE;
	typedef RefCompo2Type<T1,T2> REF_DATA_TYPE;

	Face2Attributes(FaceAttribute<T1, MAP>& h1, FaceAttribute<T2, MAP>& h2):
		m_h1(h1), m_h2(h2) {}

	RefCompo2Type<T1,T2> operator[](unsigned int a)
	{
		return RefCompo2Type<T1,T2>(m_h1[a],m_h2[a]);
	}

	RefCompo2Type<T1,T2> operator[](Face d)
	{
		return RefCompo2Type<T1,T2>(m_h1[d],m_h2[d]);
	}

	const RefCompo2Type<T1,T2> operator[](unsigned int a) const
	{
		return RefCompo2Type<T1,T2>(m_h1[a],m_h2[a]);
	}

	const RefCompo2Type<T1,T2> operator[](Face d) const
	{
		return RefCompo2Type<T1,T2>(m_h1[d],m_h2[d]);
	}

	static unsigned int getOrbit() { return FACE; }
};

template <typename T1, typename T2, typename MAP>
class Volume2Attributes
{
	VolumeAttribute<T1, MAP>& m_h1;
	VolumeAttribute<T2, MAP>& m_h2;
public:
	typedef Compo2Type<T1,T2> DATA_TYPE;
	typedef RefCompo2Type<T1,T2> REF_DATA_TYPE;

	 Volume2Attributes(VolumeAttribute<T1, MAP>& h1, VolumeAttribute<T2, MAP>& h2):
		m_h1(h1), m_h2(h2) {}

	RefCompo2Type<T1,T2> operator[](unsigned int a)
	{
		return RefCompo2Type<T1,T2>(m_h1[a],m_h2[a]);
	}

	RefCompo2Type<T1,T2> operator[](Vol d)
	{
		return RefCompo2Type<T1,T2>(m_h1[d],m_h2[d]);
	}

	const RefCompo2Type<T1,T2> operator[](unsigned int a) const
	{
		return RefCompo2Type<T1,T2>(m_h1[a],m_h2[a]);
	}

	const RefCompo2Type<T1,T2> operator[](Vol d) const
	{
		return RefCompo2Type<T1,T2>(m_h1[d],m_h2[d]);
	}

	static unsigned int getOrbit() { return VOLUME; }

};


/// implementation

template < typename T1, typename T2>
Compo2Type<T1,T2>::Compo2Type(const RefCompo2Type<T1,T2>& v)
{
	m_v1 = v.m_v1;
	m_v2 = v.m_v2;
}

template < typename T1, typename T2>
Compo2Type<T1,T2>& Compo2Type<T1,T2>::operator =(const Compo2Type<T1,T2>& comp)
{
	m_v1 = comp.m_v1;
	m_v2 = comp.m_v2;
	return *this;
}

template < typename T1, typename T2>
Compo2Type<T1,T2> Compo2Type<T1,T2>::operator+(const Compo2Type<T1,T2>& v) const
{
	Compo2Type<T1,T2> res ;
	res.m_v1 = this->m_v1 + v.m_v1;
	res.m_v2 = this->m_v2 + v.m_v2;
	return res ;
}

template < typename T1, typename T2>
Compo2Type<T1,T2> Compo2Type<T1,T2>::operator-(const Compo2Type<T1,T2>& v) const
{
	Compo2Type<T1,T2> res ;
	res.m_v1 = this->m_v1 - v.m_v1;
	res.m_v2 = this->m_v2 - v.m_v2;
	return res ;
}

template < typename T1, typename T2>
Compo2Type<T1,T2> Compo2Type<T1,T2>::operator/(double d) const
{
	Compo2Type<T1,T2> res ;
	res.m_v1 = this->m_v1 / d;
	res.m_v2 = this->m_v2 / d;
	return res ;
}

template < typename T1, typename T2>
Compo2Type<T1,T2> Compo2Type<T1,T2>::operator*(double d) const
{
	Compo2Type<T1,T2> res ;
	res.m_v1 = this->m_v1 * d;
	res.m_v2 = this->m_v2 * d;
	return res ;
}

template < typename T1, typename T2>
Compo2Type<T1,T2>& Compo2Type<T1,T2>::operator+=(const Compo2Type<T1,T2>& v)
{
	m_v1 += v.m_v1;
	m_v2 += v.m_v2;
	return *this;
}

template < typename T1, typename T2>
Compo2Type<T1,T2>& Compo2Type<T1,T2>::operator-=(const Compo2Type<T1,T2>& v)
{
	m_v1 -= v.m_v1;
	m_v2 -= v.m_v2;
	return *this;
}

template < typename T1, typename T2>
Compo2Type<T1,T2>& Compo2Type<T1,T2>::operator*=(double d)
{
	m_v1 *= d;
	m_v2 *= d;
	return *this;
}

template < typename T1, typename T2>
Compo2Type<T1,T2>& Compo2Type<T1,T2>::operator/=(double d)
{
	m_v1 /= d;
	m_v2 /= d;
	return *this;
}

/// Ref version

template < typename T1, typename T2>
RefCompo2Type<T1,T2>::RefCompo2Type (Compo2Type<T1,T2>& comp):
	m_v1(comp.m_v1),
	m_v2(comp.m_v2)
{
}

template < typename T1, typename T2>
RefCompo2Type<T1,T2>& RefCompo2Type<T1,T2>::operator=(const RefCompo2Type<T1,T2>& v)
{
	m_v1 = v.m_v1;
	m_v2 = v.m_v2;
	return *this;
}

template < typename T1, typename T2>
RefCompo2Type<T1,T2>& RefCompo2Type<T1,T2>::operator=(Compo2Type<T1,T2>  v)
{
	m_v1 = v.m_v1;
	m_v2 = v.m_v2;
	return *this;
}

template < typename T1, typename T2>
Compo2Type<T1,T2> RefCompo2Type<T1,T2>::operator+(const RefCompo2Type<T1,T2>& v) const
{
	Compo2Type<T1,T2> res ;
	res.m_v1 = this->m_v1 + v.m_v1;
	res.m_v2 = this->m_v2 + v.m_v2;
	return res ;
}

template < typename T1, typename T2>
Compo2Type<T1,T2> RefCompo2Type<T1,T2>::operator-(const RefCompo2Type<T1,T2>& v) const
{
	Compo2Type<T1,T2> res ;
	res.m_v1 = this->m_v1 - v.m_v1;
	res.m_v2 = this->m_v2 - v.m_v2;
	return res ;
}

template < typename T1, typename T2>
Compo2Type<T1,T2> RefCompo2Type<T1,T2>::operator/(double d) const
{
	Compo2Type<T1,T2> res ;
	res.m_v1 = this->m_v1 / d;
	res.m_v2 = this->m_v2 / d;
	return res ;
}

template < typename T1, typename T2>
Compo2Type<T1,T2> RefCompo2Type<T1,T2>::operator*(double d) const
{
	Compo2Type<T1,T2> res ;
	res.m_v1 = this->m_v1 * d;
	res.m_v2 = this->m_v2 * d;
	return res ;
}

template < typename T1, typename T2>
RefCompo2Type<T1,T2>& RefCompo2Type<T1,T2>::operator+=(const RefCompo2Type<T1,T2>& v)
{
	m_v1 += v.m_v1;
	m_v2 += v.m_v2;
	return *this;
}

template < typename T1, typename T2>
RefCompo2Type<T1,T2>& RefCompo2Type<T1,T2>::operator-=(const RefCompo2Type<T1,T2>& v)
{
	m_v1 -= v.m_v1;
	m_v2 -= v.m_v2;
	return *this;
}

template < typename T1, typename T2>
RefCompo2Type<T1,T2>& RefCompo2Type<T1,T2>::operator*=(double d)
{
	m_v1 *= d;
	m_v2 *= d;
	return *this;
}

template < typename T1, typename T2>
RefCompo2Type<T1,T2>& RefCompo2Type<T1,T2>::operator/=(double d)
{
	m_v1 /= d;
	m_v2 /= d;
	return *this;
}

} // namespace CGoGN

#endif /* MULTIATTRIBS_H_ */
