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

#ifndef __MULTI3ATTRIBS_H_
#define __MULTI3ATTRIBS_H_

#include "Topology/generic/cells.h"

namespace CGoGN
{

//forward
template <typename T1, typename T2, typename T3>
struct RefCompo3Type;


template <typename T1, typename T2, typename T3>
struct Compo3Type
{
	T1 m_v1;
	T2 m_v2;
	T3 m_v3;


	Compo3Type(){}
	Compo3Type(double z ): m_v1(z), m_v2(z), m_v3(z) {}
	Compo3Type(const RefCompo3Type<T1,T2,T3>& v);

	Compo3Type<T1,T2,T3>& operator =(const Compo3Type<T1,T2,T3>& comp);

	Compo3Type<T1,T2,T3> operator+(const Compo3Type<T1,T2,T3>& v) const;
	Compo3Type<T1,T2,T3> operator-(const Compo3Type<T1,T2,T3>& v) const;
	Compo3Type<T1,T2,T3> operator/(double d) const;
	Compo3Type<T1,T2,T3> operator*(double d) const;
	Compo3Type<T1,T2,T3>& operator+=(const Compo3Type<T1,T2,T3>& v);
	Compo3Type<T1,T2,T3>& operator-=(const Compo3Type<T1,T2,T3>& v);
	Compo3Type<T1,T2,T3>& operator*=(double d);
	Compo3Type<T1,T2,T3>& operator/=(double d);

};


template <typename T1, typename T2, typename T3>
struct RefCompo3Type
{
	T1& m_v1;
	T2& m_v2;
	T3& m_v3;

	RefCompo3Type(T1& v1, T2& v2, T3& v3): m_v1(v1), m_v2(v2), m_v3(v3) {}
	RefCompo3Type (Compo3Type<T1,T2,T3>& comp);

	RefCompo3Type<T1,T2,T3>& operator=(const RefCompo3Type<T1,T2,T3>& v);
	RefCompo3Type<T1,T2,T3>& operator=(Compo3Type<T1,T2,T3> v);

	Compo3Type<T1,T2,T3> operator+(const RefCompo3Type<T1,T2,T3>& v) const;
	Compo3Type<T1,T2,T3> operator-(const RefCompo3Type<T1,T2,T3>& v) const;
	Compo3Type<T1,T2,T3> operator/(double d) const;
	Compo3Type<T1,T2,T3> operator*(double d) const;
	RefCompo3Type<T1,T2,T3>& operator+=(const RefCompo3Type<T1,T2,T3>& v);
	RefCompo3Type<T1,T2,T3>& operator-=(const RefCompo3Type<T1,T2,T3>& v);
	RefCompo3Type<T1,T2,T3>& operator*=(double d);
	RefCompo3Type<T1,T2,T3>& operator/=(double d);
};



template <typename T1, typename T2,  typename T3>
class Vertex3Attributes
{
	VertexAttribute<T1>& m_h1;
	VertexAttribute<T2>& m_h2;
	VertexAttribute<T3>& m_h3;
public:
	typedef Compo3Type<T1,T2,T3> DATA_TYPE;
	typedef RefCompo3Type<T1,T2,T3> REF_DATA_TYPE;

	Vertex3Attributes(VertexAttribute<T1>& h1, VertexAttribute<T2>& h2, VertexAttribute<T3>& h3):
		m_h1(h1), m_h2(h2), m_h3(h3) {}

	RefCompo3Type<T1,T2,T3> operator[](unsigned int a)
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[a],m_h2[a],m_h3[a]);
	}

	RefCompo3Type<T1,T2,T3> operator[](Vertex d)
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[d],m_h2[d],m_h3[d]);
	}

	const RefCompo3Type<T1,T2,T3> operator[](unsigned int a) const
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[a],m_h2[a],m_h3[a]);
	}

	const RefCompo3Type<T1,T2,T3> operator[](Vertex d) const
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[d],m_h2[d],m_h3[d]);
	}

	static unsigned int getOrbit() { return VERTEX; }
};


template <typename T1, typename T2,  typename T3>
class Edge3Attributes
{
	EdgeAttribute<T1>& m_h1;
	EdgeAttribute<T2>& m_h2;
	EdgeAttribute<T3>& m_h3;
public:
	typedef Compo3Type<T1,T2,T3> DATA_TYPE;
	typedef RefCompo3Type<T1,T2,T3> REF_DATA_TYPE;

	Edge3Attributes(EdgeAttribute<T1>& h1, EdgeAttribute<T2>& h2, EdgeAttribute<T3>& h3):
		m_h1(h1), m_h2(h2), m_h3(h3) {}

	RefCompo3Type<T1,T2,T3> operator[](unsigned int a)
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[a],m_h2[a],m_h3[a]);
	}

	RefCompo3Type<T1,T2,T3> operator[](Edge d)
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[d],m_h2[d],m_h3[d]);
	}

	const RefCompo3Type<T1,T2,T3> operator[](unsigned int a) const
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[a],m_h2[a],m_h3[a]);
	}

	const RefCompo3Type<T1,T2,T3> operator[](Edge d) const
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[d],m_h2[d],m_h3[d]);
	}

	static unsigned int getOrbit() { return EDGE; }
};



template <typename T1, typename T2,  typename T3>
class Face3Attributes
{
	FaceAttribute<T1>& m_h1;
	FaceAttribute<T2>& m_h2;
	FaceAttribute<T3>& m_h3;
public:
	typedef Compo3Type<T1,T2,T3> DATA_TYPE;
	typedef RefCompo3Type<T1,T2,T3> REF_DATA_TYPE;

	Face3Attributes(FaceAttribute<T1>& h1, FaceAttribute<T2>& h2, FaceAttribute<T3>& h3):
		m_h1(h1), m_h2(h2), m_h3(h3) {}

	RefCompo3Type<T1,T2,T3> operator[](unsigned int a)
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[a],m_h2[a],m_h3[a]);
	}

	RefCompo3Type<T1,T2,T3> operator[](Face d)
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[d],m_h2[d],m_h3[d]);
	}

	const RefCompo3Type<T1,T2,T3> operator[](unsigned int a) const
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[a],m_h2[a],m_h3[a]);
	}

	const RefCompo3Type<T1,T2,T3> operator[](Face d) const
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[d],m_h2[d],m_h3[d]);
	}

	static unsigned int getOrbit() { return FACE; }
};

template <typename T1, typename T2,  typename T3>
class Volume3Attributes
{
	VolumeAttribute<T1>& m_h1;
	VolumeAttribute<T2>& m_h2;
	VolumeAttribute<T3>& m_h3;
public:
	typedef Compo3Type<T1,T2,T3> DATA_TYPE;
	typedef RefCompo3Type<T1,T2,T3> REF_DATA_TYPE;

	 Volume3Attributes(VolumeAttribute<T1>& h1, VolumeAttribute<T2>& h2, VolumeAttribute<T2>& h3):
		 m_h1(h1), m_h2(h2), m_h3(h3) {}

	RefCompo3Type<T1,T2,T3> operator[](unsigned int a)
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[a],m_h2[a],m_h3[a]);
	}

	RefCompo3Type<T1,T2,T3> operator[](Vol d)
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[d],m_h2[d],m_h3[d]);
	}

	const RefCompo3Type<T1,T2,T3> operator[](unsigned int a) const
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[a],m_h2[a],m_h3[a]);
	}

	const RefCompo3Type<T1,T2,T3> operator[](Vol d) const
	{
		return RefCompo3Type<T1,T2,T3>(m_h1[d],m_h2[d],m_h3[d]);
	}

	static unsigned int getOrbit() { return VOLUME; }

};



/// implementation

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3>& Compo3Type<T1,T2,T3>::operator =(const Compo3Type<T1,T2,T3>& comp)
{
	m_v1 = comp.m_v1;
	m_v2 = comp.m_v2;
	m_v3 = comp.m_v3;
	return *this;
}


template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3>::Compo3Type(const RefCompo3Type<T1, T2, T3> &v)
{
	m_v1 = v.m_v1;
	m_v2 = v.m_v2;
	m_v3 = v.m_v3;
}

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3> Compo3Type<T1,T2,T3>::operator+(const Compo3Type<T1,T2,T3>& v) const
{
	Compo3Type<T1,T2,T3> res ;
	res.m_v1 = this->m_v1 + v.m_v1;
	res.m_v2 = this->m_v2 + v.m_v2;
	res.m_v3 = this->m_v3 + v.m_v3;
	return res ;
}



template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3> Compo3Type<T1,T2,T3>::operator-(const Compo3Type<T1,T2,T3>& v) const
{
	Compo3Type<T1,T2,T3> res ;
	res.m_v1 = this->m_v1 - v.m_v1;
	res.m_v2 = this->m_v2 - v.m_v2;
	res.m_v3 = this->m_v3 - v.m_v3;
	return res ;
}

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3> Compo3Type<T1,T2,T3>::operator/(double d) const
{
	Compo3Type<T1,T2,T3> res ;
	res.m_v1 = this->m_v1 / d;
	res.m_v2 = this->m_v2 / d;
	res.m_v3 = this->m_v3 / d;
	return res ;
}

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3> Compo3Type<T1,T2,T3>::operator*(double d) const
{
	Compo3Type<T1,T2,T3> res ;
	res.m_v1 = this->m_v1 * d;
	res.m_v2 = this->m_v2 * d;
	res.m_v3 = this->m_v3 * d;
		return res ;
}

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3>& Compo3Type<T1,T2,T3>::operator+=(const Compo3Type<T1,T2,T3>& v)
{
	m_v1 += v.m_v1;
	m_v2 += v.m_v2;
	m_v3 += v.m_v3;
	return *this;
}

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3>& Compo3Type<T1,T2,T3>::operator-=(const Compo3Type<T1,T2,T3>& v)
{
	m_v1 += v.m_v1;
	m_v2 -= v.m_v2;
	m_v3 -= v.m_v3;
	return *this;
}

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3>& Compo3Type<T1,T2,T3>::operator*=(double d)
{
	m_v1 *= d;
	m_v2 *= d;
	m_v3 *= d;
	return *this;
}

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3>& Compo3Type<T1,T2,T3>::operator/=(double d)
{
	m_v1 /= d;
	m_v2 /= d;
	m_v3 /= d;
	return *this;
}


/// Ref version

template < typename T1, typename T2, typename T3>
RefCompo3Type<T1,T2,T3>::RefCompo3Type (Compo3Type<T1,T2,T3>& comp):
	m_v1(comp.m_v1),
	m_v2(comp.m_v2),
	m_v3(comp.m_v3)
{
}

template < typename T1, typename T2, typename T3>
RefCompo3Type<T1,T2,T3>& RefCompo3Type<T1,T2,T3>::operator=(const RefCompo3Type<T1,T2,T3>& v)
{
	m_v1 = v.m_v1;
	m_v2 = v.m_v2;
	m_v3 = v.m_v3;
	return *this;
}

template < typename T1, typename T2, typename T3>
RefCompo3Type<T1,T2,T3>& RefCompo3Type<T1,T2,T3>::operator=(Compo3Type<T1,T2,T3> v)
{
	m_v1 = v.m_v1;
	m_v2 = v.m_v2;
	m_v3 = v.m_v3;
	return *this;
}


template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3> RefCompo3Type<T1,T2,T3>::operator+(const RefCompo3Type<T1,T2,T3>& v) const
{
	Compo3Type<T1,T2,T3> res ;
	res.m_v1 = this->m_v1 + v.m_v1;
	res.m_v2 = this->m_v2 + v.m_v2;
	res.m_v3 = this->m_v3 + v.m_v3;
	return res ;
}

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3> RefCompo3Type<T1,T2,T3>::operator-(const RefCompo3Type<T1,T2,T3>& v) const
{
	Compo3Type<T1,T2,T3> res ;
	res.m_v1 = this->m_v1 - v.m_v1;
	res.m_v2 = this->m_v2 - v.m_v2;
	res.m_v3 = this->m_v3 - v.m_v3;
	return res ;
}

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3> RefCompo3Type<T1,T2,T3>::operator/(double d) const
{
	Compo3Type<T1,T2,T3> res ;
	res.m_v1 = this->m_v1 / d;
	res.m_v2 = this->m_v2 / d;
	res.m_v3 = this->m_v3 / d;
	return res ;
}

template < typename T1, typename T2, typename T3>
Compo3Type<T1,T2,T3> RefCompo3Type<T1,T2,T3>::operator*(double d) const
{
	Compo3Type<T1,T2,T3> res ;
	res.m_v1 = this->m_v1 * d;
	res.m_v2 = this->m_v2 * d;
	res.m_v3 = this->m_v3 * d;
	return res ;
}

template < typename T1, typename T2, typename T3>
RefCompo3Type<T1,T2,T3>& RefCompo3Type<T1,T2,T3>::operator+=(const RefCompo3Type<T1,T2,T3>& v)
{
	m_v1 += v.m_v1;
	m_v2 += v.m_v2;
	m_v3 += v.m_v3;
	return *this;
}

template < typename T1, typename T2, typename T3>
RefCompo3Type<T1,T2,T3>& RefCompo3Type<T1,T2,T3>::operator-=(const RefCompo3Type<T1,T2,T3>& v)
{
	m_v1 -= v.m_v1;
	m_v2 -= v.m_v2;
	m_v3 -= v.m_v3;
	return *this;
}

template < typename T1, typename T2, typename T3>
RefCompo3Type<T1,T2,T3>& RefCompo3Type<T1,T2,T3>::operator*=(double d)
{
	m_v1 *= d;
	m_v2 *= d;
	m_v3 *= d;
	return *this;
}

template < typename T1, typename T2, typename T3>
RefCompo3Type<T1,T2,T3>& RefCompo3Type<T1,T2,T3>::operator/=(double d)
{
	m_v1 /= d;
	m_v2 /= d;
	m_v3 /= d;
	return *this;
}


} // namespace CGoGN

#endif /* MULTIATTRIBS_H_ */
