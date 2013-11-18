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

#ifndef __FIXED_ARRAY__
#define __FIXED_ARRAY__

#include <cassert>
#include <iostream>
#include <math.h>
#include <string.h>

#include "Utils/static_assert.h"
#include "Utils/nameTypes.h"

namespace CGoGN
{

namespace Utils
{

/*
 * Class for the representation of vectors
 */
template <unsigned int DIM, typename T>
class FixedArray
{
protected:
	T m_data[DIM] ;
public:
	typedef T DATA_TYPE ;

	static std::string CGoGNnameOfType() ;

	/**********************************************/
	/*                CONSTRUCTORS                */
	/**********************************************/

	FixedArray() ;

	/**
	 * copy constructor 
	 * @param a the array
	 */
	FixedArray(const FixedArray<DIM, T>& v) ;

	/**
	 * constructor that initialize all component to a given value
	 * @param x the value to assign to all component
	 */
	template <typename T2>
	FixedArray(const FixedArray<DIM, T2>& v) ;


	template <unsigned int DIM2>
	FixedArray(const FixedArray<DIM2, T>& v) ;


	template <unsigned int DIM2, typename T2>
	FixedArray(const FixedArray<DIM2, T2>& v) ;


	template <typename T2>
	FixedArray<DIM,T>& operator=(const FixedArray<DIM, T2>& v) ;


	template <unsigned int DIM2>
	FixedArray<DIM,T>& operator=(const FixedArray<DIM2, T>& v) ;


	template <unsigned int DIM2, typename T2>
	FixedArray<DIM,T>& operator=(const FixedArray<DIM2, T2>& v);


	/**
	 * constructor that initialize all component to a given value
	 * @param x the value to assign to all component
	 */
	FixedArray(T x);

	/**
	 * @brief affect same value to all array
	 * @param x value
	 */
	void set(T x);

	/**
	 * @brief operator []
	 * @param index
	 * @return value
	 */
	T& operator[](unsigned int index);

	/**
	 * @brief const operator []
	 * @param index
	 * @return value
	 */
	const T& operator[](unsigned int index) const;

	/**
	 * @brief get access to data ptr
	 * @return ptr
	 */
	T* data();

	/**
	 * @brief get access to data ptr
	 * @return ptr
	 */
	const T* data() const;

	/**
	 * @brief get dimension of array (DIM template param)
	 * @return DIM
	 */
	static unsigned int dimension() {return DIM;}
};



template <unsigned int DIM, typename T>
inline FixedArray<DIM,T>::FixedArray()
{}

template <unsigned int DIM, typename T>
inline FixedArray<DIM,T>::FixedArray(const FixedArray<DIM, T>& v)
{
	for (unsigned int i=0; i< DIM; ++i)
		m_data[i] = v.m_data[i];
}

template <unsigned int DIM, typename T>
template <typename T2>
inline FixedArray<DIM,T>::FixedArray(const FixedArray<DIM, T2>& v)
{
	for (unsigned int i=0; i< DIM; ++i)
		m_data[i] = T(v[i]);
}

template <unsigned int DIM, typename T>
template <unsigned int DIM2>
inline FixedArray<DIM,T>::FixedArray(const FixedArray<DIM2, T>& v)
{
	if (DIM2 <= DIM)
		for (unsigned int i=0; i< DIM2; ++i)
			m_data[i] = v[i];
	else
		for (unsigned int i=0; i< DIM; ++i)
			m_data[i] = v[i];
}


template <unsigned int DIM, typename T>
template <unsigned int DIM2, typename T2>
inline FixedArray<DIM,T>::FixedArray(const FixedArray<DIM2, T2>& v)
{
	if (DIM2 <= DIM)
		for (unsigned int i=0; i< DIM2; ++i)
			m_data[i] = T(v[i]);
	else
		for (unsigned int i=0; i< DIM; ++i)
			m_data[i] = T(v[i]);
}


template <unsigned int DIM, typename T>
template <typename T2>
inline FixedArray<DIM,T>& FixedArray<DIM,T>::operator=(const FixedArray<DIM, T2>& v)
{
	for (unsigned int i=0; i< DIM; ++i)
		m_data[i] = T(v[i]);
	return *this;
}


template <unsigned int DIM, typename T>
template <unsigned int DIM2>
inline FixedArray<DIM,T>& FixedArray<DIM,T>::operator=(const FixedArray<DIM2, T>& v)
{
	if (DIM2 <= DIM)
		for (unsigned int i=0; i< DIM2; ++i)
			m_data[i] = v[i];
	else
		for (unsigned int i=0; i< DIM; ++i)
			m_data[i] = v[i];
	return *this;
}


template <unsigned int DIM, typename T>
template <unsigned int DIM2, typename T2>
inline FixedArray<DIM,T>& FixedArray<DIM,T>::operator=(const FixedArray<DIM2, T2>& v)
{
	if (DIM2 <= DIM)
		for (unsigned int i=0; i< DIM2; ++i)
			m_data[i] = T(v[i]);
	else
		for (unsigned int i=0; i< DIM; ++i)
			m_data[i] = T(v[i]);
	return *this;
}




template <unsigned int DIM, typename T>
inline FixedArray<DIM,T>::FixedArray(T x)
{
	for (unsigned int i=0; i< DIM; ++i)
		m_data[i] = x;
}


template <unsigned int DIM, typename T>
inline void FixedArray<DIM,T>::set(T x)
{
	for (unsigned int i = 0; i < DIM; ++i)
		m_data[i] = x ;
}

template <unsigned int DIM, typename T>
inline T& FixedArray<DIM,T>::operator[](unsigned int index)
{
	assert(index < DIM) ;
	return m_data[index];
}

template <unsigned int DIM, typename T>
inline const T& FixedArray<DIM,T>::operator[](unsigned int index) const
{
	assert(index < DIM) ;
	return m_data[index];
}

template <unsigned int DIM, typename T>
inline T* FixedArray<DIM,T>::data()
{
	return m_data;
}

template <unsigned int DIM, typename T>
inline const T* FixedArray<DIM,T>::data() const
{
	return m_data;
}

template <unsigned int DIM, typename T>
std::string FixedArray<DIM, T>::CGoGNnameOfType()
{
//	std::stringstream ss ;
//	ss << "Utils::FixedArray<" ;
//	ss << DIM ;
//	ss << "," ;
//	ss << nameOfType(T()) ;
//	ss << ">" ;

//	return ss.str() ;
	return "FixedArray<?>";
}

}
}
#endif

