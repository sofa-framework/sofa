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

#ifndef __MULTI4ATTRIBS_H_
#define __MULTI4ATTRIBS_H_

#include "Topology/generic/cells.h"

namespace CGoGN
{

//forward
template <typename T1, typename T2, typename T3, typename T4>
struct RefCompo4Type;


template <typename T1, typename T2, typename T3, typename T4>
struct Compo4Type
{
	T1 m_v1;
	T2 m_v2;
	T3 m_v3;
	T4 m_v4;


	Compo4Type(){}
	Compo4Type(double z ): m_v1(z), m_v2(z), m_v3(z), m_v4(z) {}
	Compo4Type(const RefCompo4Type<T1,T2,T3,T4>& v);

	Compo4Type<T1,T2,T3,T4>& operator =(const Compo4Type<T1,T2,T3,T4>& comp);

	Compo4Type<T1,T2,T3,T4> operator+(const Compo4Type<T1,T2,T3,T4>& v) const;
	Compo4Type<T1,T2,T3,T4> operator-(const Compo4Type<T1,T2,T3,T4>& v) const;
	Compo4Type<T1,T2,T3,T4> operator/(double d) const;
	Compo4Type<T1,T2,T3,T4> operator*(double d) const;
	Compo4Type<T1,T2,T3,T4>& operator+=(const Compo4Type<T1,T2,T3,T4>& v);
	Compo4Type<T1,T2,T3,T4>& operator-=(const Compo4Type<T1,T2,T3,T4>& v);
	Compo4Type<T1,T2,T3,T4>& operator*=(double d);
	Compo4Type<T1,T2,T3,T4>& operator/=(double d);

};


template <typename T1, typename T2, typename T3, typename T4>
struct RefCompo4Type
{
	T1& m_v1;
	T2& m_v2;
	T3& m_v3;
	T4& m_v4;

	RefCompo4Type(T1& v1, T2& v2, T3& v3, T4& v4): m_v1(v1), m_v2(v2), m_v3(v3), m_v4(v4) {}
	RefCompo4Type (Compo4Type<T1,T2,T3,T4>& comp);

	RefCompo4Type<T1,T2,T3,T4>& operator=(const RefCompo4Type<T1,T2,T3,T4>& v);
	RefCompo4Type<T1,T2,T3,T4>& operator=(Compo4Type<T1,T2,T3,T4> v);

	Compo4Type<T1,T2,T3,T4> operator+(const RefCompo4Type<T1,T2,T3,T4>& v) const;
	Compo4Type<T1,T2,T3,T4> operator-(const RefCompo4Type<T1,T2,T3,T4>& v) const;
	Compo4Type<T1,T2,T3,T4> operator/(double d) const;
	Compo4Type<T1,T2,T3,T4> operator*(double d) const;
	RefCompo4Type<T1,T2,T3,T4>& operator+=(const RefCompo4Type<T1,T2,T3,T4>& v);
	RefCompo4Type<T1,T2,T3,T4>& operator-=(const RefCompo4Type<T1,T2,T3,T4>& v);
	RefCompo4Type<T1,T2,T3,T4>& operator*=(double d);
	RefCompo4Type<T1,T2,T3,T4>& operator/=(double d);
};



template <typename T1, typename T2,  typename T3, typename T4>
class Vertex4Attributes
{
	VertexAttribute<T1>& m_h1;
	VertexAttribute<T2>& m_h2;
	VertexAttribute<T3>& m_h3;
	VertexAttribute<T4>& m_h4;
public:
	typedef Compo4Type<T1,T2,T3,T4> DATA_TYPE;
	typedef RefCompo4Type<T1,T2,T3,T4> REF_DATA_TYPE;

	Vertex4Attributes(VertexAttribute<T1>& h1, VertexAttribute<T2>& h2, VertexAttribute<T3>& h3, VertexAttribute<T4>& h4):
		m_h1(h1), m_h2(h2), m_h3(h3), m_h4(h4) {}

	RefCompo4Type<T1,T2,T3,T4> operator[](unsigned int a)
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[a],m_h2[a],m_h3[a],m_h4[a]);
	}

	RefCompo4Type<T1,T2,T3,T4> operator[](Vertex d)
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[d],m_h2[d],m_h3[d],m_h4[d]);
	}

	const RefCompo4Type<T1,T2,T3,T4> operator[](unsigned int a) const
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[a],m_h2[a],m_h3[a],m_h4[a]);
	}

	const RefCompo4Type<T1,T2,T3,T4> operator[](Vertex d) const
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[d],m_h2[d],m_h3[d],m_h4[d]);
	}

	static unsigned int getOrbit() { return VERTEX; }
};


template <typename T1, typename T2,  typename T3, typename T4>
class Edge4Attributes
{
	EdgeAttribute<T1>& m_h1;
	EdgeAttribute<T2>& m_h2;
	EdgeAttribute<T3>& m_h3;
	EdgeAttribute<T4>& m_h4;
public:
	typedef Compo4Type<T1,T2,T3,T4> DATA_TYPE;
	typedef RefCompo4Type<T1,T2,T3,T4> REF_DATA_TYPE;

	Edge4Attributes(EdgeAttribute<T1>& h1, EdgeAttribute<T2>& h2, EdgeAttribute<T3>& h3, EdgeAttribute<T4>& h4):
		m_h1(h1), m_h2(h2), m_h3(h3), m_h4(h4) {}

	RefCompo4Type<T1,T2,T3,T4> operator[](unsigned int a)
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[a],m_h2[a],m_h3[a],m_h4[a]);
	}

	RefCompo4Type<T1,T2,T3,T4> operator[](Edge d)
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[d],m_h2[d],m_h3[d],m_h4[d]);
	}

	const RefCompo4Type<T1,T2,T3,T4> operator[](unsigned int a) const
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[a],m_h2[a],m_h3[a],m_h4[a]);
	}

	const RefCompo4Type<T1,T2,T3,T4> operator[](Edge d) const
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[d],m_h2[d],m_h3[d],m_h4[d]);
	}

	static unsigned int getOrbit() { return EDGE; }
};



template <typename T1, typename T2,  typename T3, typename T4>
class Face4Attributes
{
	FaceAttribute<T1>& m_h1;
	FaceAttribute<T2>& m_h2;
	FaceAttribute<T3>& m_h3;
	FaceAttribute<T4>& m_h4;
public:
	typedef Compo4Type<T1,T2,T3,T4> DATA_TYPE;
	typedef RefCompo4Type<T1,T2,T3,T4> REF_DATA_TYPE;

	Face4Attributes(FaceAttribute<T1>& h1, FaceAttribute<T2>& h2, FaceAttribute<T3>& h3, FaceAttribute<T4>& h4):
		m_h1(h1), m_h2(h2), m_h3(h3), m_h4(h4) {}

	RefCompo4Type<T1,T2,T3,T4> operator[](unsigned int a)
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[a],m_h2[a],m_h3[a],m_h4[a]);
	}

	RefCompo4Type<T1,T2,T3,T4> operator[](Face d)
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[d],m_h2[d],m_h3[d],m_h4[d]);
	}

	const RefCompo4Type<T1,T2,T3,T4> operator[](unsigned int a) const
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[a],m_h2[a],m_h3[a],m_h4[a]);
	}

	const RefCompo4Type<T1,T2,T3,T4> operator[](Face d) const
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[d],m_h2[d],m_h3[d],m_h4[d]);
	}

	static unsigned int getOrbit() { return FACE; }
};

template <typename T1, typename T2,  typename T3, typename T4>
class Volume4Attributes
{
	VolumeAttribute<T1>& m_h1;
	VolumeAttribute<T2>& m_h2;
	VolumeAttribute<T3>& m_h3;
	VolumeAttribute<T4>& m_h4;
public:
	typedef Compo4Type<T1,T2,T3,T4> DATA_TYPE;
	typedef RefCompo4Type<T1,T2,T3,T4> REF_DATA_TYPE;

	 Volume4Attributes(VolumeAttribute<T1>& h1, VolumeAttribute<T2>& h2, VolumeAttribute<T3>& h3, VolumeAttribute<T4>& h4):
		 m_h1(h1), m_h2(h2), m_h3(h3), m_h4(h4) {}

	RefCompo4Type<T1,T2,T3,T4> operator[](unsigned int a)
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[a],m_h2[a],m_h3[a],m_h4[a]);
	}

	RefCompo4Type<T1,T2,T3,T4> operator[](Vol d)
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[d],m_h2[d],m_h3[d],m_h4[d]);
	}

	const RefCompo4Type<T1,T2,T3,T4> operator[](unsigned int a) const
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[a],m_h2[a],m_h3[a],m_h4[a]);
	}

	const RefCompo4Type<T1,T2,T3,T4> operator[](Vol d) const
	{
		return RefCompo4Type<T1,T2,T3,T4>(m_h1[d],m_h2[d],m_h3[d],m_h4[d]);
	}

	static unsigned int getOrbit() { return VOLUME; }
};



/// implementation

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4>& Compo4Type<T1,T2,T3,T4>::operator =(const Compo4Type<T1,T2,T3,T4>& comp)
{
	m_v1 = comp.m_v1;
	m_v2 = comp.m_v2;
	m_v3 = comp.m_v3;
	m_v4 = comp.m_v4;
	return *this;
}


template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4>::Compo4Type(const RefCompo4Type<T1,T2,T3,T4> &v)
{
	m_v1 = v.m_v1;
	m_v2 = v.m_v2;
	m_v3 = v.m_v3;
	m_v4 = v.m_v4;
}

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4> Compo4Type<T1,T2,T3,T4>::operator+(const Compo4Type<T1,T2,T3,T4>& v) const
{
	Compo4Type<T1,T2,T3,T4> res ;
	res.m_v1 = this->m_v1 + v.m_v1;
	res.m_v2 = this->m_v2 + v.m_v2;
	res.m_v3 = this->m_v3 + v.m_v3;
	res.m_v4 = this->m_v4 + v.m_v4;
	return res ;
}



template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4> Compo4Type<T1,T2,T3,T4>::operator-(const Compo4Type<T1,T2,T3,T4>& v) const
{
	Compo4Type<T1,T2,T3,T4> res ;
	res.m_v1 = this->m_v1 - v.m_v1;
	res.m_v2 = this->m_v2 - v.m_v2;
	res.m_v3 = this->m_v3 - v.m_v3;
	res.m_v4 = this->m_v4 - v.m_v4;
	return res ;
}

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4> Compo4Type<T1,T2,T3,T4>::operator/(double d) const
{
	Compo4Type<T1,T2,T3,T4> res ;
	res.m_v1 = this->m_v1 / d;
	res.m_v2 = this->m_v2 / d;
	res.m_v3 = this->m_v3 / d;
	res.m_v4 = this->m_v4 / d;
	return res ;
}

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4> Compo4Type<T1,T2,T3,T4>::operator*(double d) const
{
	Compo4Type<T1,T2,T3,T4> res ;
	res.m_v1 = this->m_v1 * d;
	res.m_v2 = this->m_v2 * d;
	res.m_v3 = this->m_v3 * d;
	res.m_v4 = this->m_v4 * d;
	return res ;
}

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4>& Compo4Type<T1,T2,T3,T4>::operator+=(const Compo4Type<T1,T2,T3,T4>& v)
{
	m_v1 += v.m_v1;
	m_v2 += v.m_v2;
	m_v3 += v.m_v3;
	m_v4 += v.m_v4;
	return *this;
}

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4>& Compo4Type<T1,T2,T3,T4>::operator-=(const Compo4Type<T1,T2,T3,T4>& v)
{
	m_v1 += v.m_v1;
	m_v2 -= v.m_v2;
	m_v3 -= v.m_v3;
	m_v4 -= v.m_v4;
	return *this;
}

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4>& Compo4Type<T1,T2,T3,T4>::operator*=(double d)
{
	m_v1 *= d;
	m_v2 *= d;
	m_v3 *= d;
	m_v4 *= d;
	return *this;
}

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4>& Compo4Type<T1,T2,T3,T4>::operator/=(double d)
{
	m_v1 /= d;
	m_v2 /= d;
	m_v3 /= d;
	m_v4 /= d;
	return *this;
}


/// Ref version

template < typename T1, typename T2, typename T3, typename T4 >
RefCompo4Type<T1,T2,T3,T4>::RefCompo4Type (Compo4Type<T1,T2,T3,T4>& comp):
	m_v1(comp.m_v1),
	m_v2(comp.m_v2),
	m_v3(comp.m_v3),
	m_v4(comp.m_v4)
{
}

template < typename T1, typename T2, typename T3, typename T4 >
RefCompo4Type<T1,T2,T3,T4>& RefCompo4Type<T1,T2,T3,T4>::operator=(const RefCompo4Type<T1,T2,T3,T4>& v)
{
	m_v1 = v.m_v1;
	m_v2 = v.m_v2;
	m_v3 = v.m_v3;
	m_v4 = v.m_v4;
	return *this;
}

template < typename T1, typename T2, typename T3, typename T4 >
RefCompo4Type<T1,T2,T3,T4>& RefCompo4Type<T1,T2,T3,T4>::operator=(Compo4Type<T1,T2,T3,T4> v)
{
	m_v1 = v.m_v1;
	m_v2 = v.m_v2;
	m_v3 = v.m_v3;
	m_v4 = v.m_v4;
	return *this;
}


template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4> RefCompo4Type<T1,T2,T3,T4>::operator+(const RefCompo4Type<T1,T2,T3,T4>& v) const
{
	Compo4Type<T1,T2,T3,T4> res ;
	res.m_v1 = this->m_v1 + v.m_v1;
	res.m_v2 = this->m_v2 + v.m_v2;
	res.m_v3 = this->m_v3 + v.m_v3;
	res.m_v4 = this->m_v4 + v.m_v4;
	return res ;
}

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4> RefCompo4Type<T1,T2,T3,T4>::operator-(const RefCompo4Type<T1,T2,T3,T4>& v) const
{
	Compo4Type<T1,T2,T3,T4> res ;
	res.m_v1 = this->m_v1 - v.m_v1;
	res.m_v2 = this->m_v2 - v.m_v2;
	res.m_v3 = this->m_v3 - v.m_v3;
	res.m_v4 = this->m_v4 - v.m_v4;
	return res ;
}

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4> RefCompo4Type<T1,T2,T3,T4>::operator/(double d) const
{
	Compo4Type<T1,T2,T3,T4> res ;
	res.m_v1 = this->m_v1 / d;
	res.m_v2 = this->m_v2 / d;
	res.m_v3 = this->m_v3 / d;
	res.m_v4 = this->m_v4 / d;

	return res ;
}

template < typename T1, typename T2, typename T3, typename T4 >
Compo4Type<T1,T2,T3,T4> RefCompo4Type<T1,T2,T3,T4>::operator*(double d) const
{
	Compo4Type<T1,T2,T3,T4> res ;
	res.m_v1 = this->m_v1 * d;
	res.m_v2 = this->m_v2 * d;
	res.m_v3 = this->m_v3 * d;
	res.m_v4 = this->m_v4 * d;
	return res ;
}

template < typename T1, typename T2, typename T3, typename T4 >
RefCompo4Type<T1,T2,T3,T4>& RefCompo4Type<T1,T2,T3,T4>::operator+=(const RefCompo4Type<T1,T2,T3,T4>& v)
{
	m_v1 += v.m_v1;
	m_v2 += v.m_v2;
	m_v3 += v.m_v3;
	m_v4 += v.m_v4;
	return *this;
}

template < typename T1, typename T2, typename T3, typename T4 >
RefCompo4Type<T1,T2,T3,T4>& RefCompo4Type<T1,T2,T3,T4>::operator-=(const RefCompo4Type<T1,T2,T3,T4>& v)
{
	m_v1 -= v.m_v1;
	m_v2 -= v.m_v2;
	m_v3 -= v.m_v3;
	m_v4 -= v.m_v4;
	return *this;
}

template < typename T1, typename T2, typename T3, typename T4 >
RefCompo4Type<T1,T2,T3,T4>& RefCompo4Type<T1,T2,T3,T4>::operator*=(double d)
{
	m_v1 *= d;
	m_v2 *= d;
	m_v3 *= d;
	m_v4 *= d;
	return *this;
}

template < typename T1, typename T2, typename T3, typename T4 >
RefCompo4Type<T1,T2,T3,T4>& RefCompo4Type<T1,T2,T3,T4>::operator/=(double d)
{
	m_v1 /= d;
	m_v2 /= d;
	m_v3 /= d;
	m_v4 /= d;
	return *this;
}


} // namespace CGoGN

#endif /* MULTIATTRIBS_H_ */
