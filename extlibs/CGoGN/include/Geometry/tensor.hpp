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

/* \file tensor.hpp */

namespace CGoGN {
namespace Geom {

template <unsigned int SIZE, typename REAL>
Tensor<SIZE, REAL>::Tensor():
m_order(0)
{
	m_data = new REAL[1] ;
}

template <unsigned int SIZE, typename REAL>
Tensor<SIZE, REAL>::Tensor(const Tensor& T):
m_order(T.m_order)
{
	m_data = new REAL[T.nbElem()] ;

	for (unsigned int i = 0 ; i < T.nbElem() ; ++i)
		m_data[i] = T[i] ;
}

template <unsigned int SIZE, typename REAL>
Tensor<SIZE, REAL>::Tensor(unsigned int order):
m_order(order)
{
	m_data = new REAL[(unsigned int)pow(SIZE,m_order)] ;
}

template <unsigned int SIZE, typename REAL>
Tensor<SIZE, REAL>::~Tensor()
{
	delete[] m_data ;
}

template <unsigned int SIZE, typename REAL>
void Tensor<SIZE, REAL>::identity()
{
	unsigned int offset = 0 ;
	for (unsigned int i = 0 ; i < SIZE ; ++i)
		m_data[SIZE*i + offset++] = 1 ;
	assert(offset == SIZE) ;
}

template <unsigned int SIZE, typename REAL>
void
Tensor<SIZE, REAL>::zero()
{
	for (unsigned int i = 0 ; i < (unsigned int)pow(SIZE,m_order) ; ++i)
	{
		m_data[i] = 0 ;
	}
}

template <unsigned int SIZE, typename REAL>
void
Tensor<SIZE, REAL>::setConst(const REAL& r)
{
	for (unsigned int i = 0 ; i < nbElem() ; ++i)
	{
		m_data[i] = r ;
	}
}

template <unsigned int SIZE, typename REAL>
void
Tensor<SIZE, REAL>::operator=(const Tensor& T)
{
	m_order = T.m_order ;

	delete[] m_data ;
	m_data = new REAL[T.nbElem()] ;

	for (unsigned int i = 0 ; i < T.nbElem() ; ++i)
		m_data[i] = T[i] ;
}

template <unsigned int SIZE, typename REAL>
const REAL&
Tensor<SIZE, REAL>::operator()(std::vector<unsigned int> p) const
{
	assert(p.size() == m_order || !"Tensor::operator(): order does not correspond to argument") ;
	return m_data[getIndex(p)] ;
}

template <unsigned int SIZE, typename REAL>
REAL&
Tensor<SIZE, REAL>::operator()(std::vector<unsigned int> p)
{
	assert(p.size() == m_order || !"Tensor::operator(): order does not correspond to argument") ;
	return m_data[getIndex(p)] ;
}

template <unsigned int SIZE, typename REAL>
unsigned int
Tensor<SIZE, REAL>::getIndex(std::vector<unsigned int> p) const
{
	assert(p.size() == m_order || !"Tensor::getIndex: order does not correspond to argument") ;
	if (p.size() != m_order)
		return -1 ;

	unsigned int res = 0 ;
	unsigned int prod = 1 ;
	for (unsigned int i = 0 ; i < m_order ; ++i)
	{
		assert(p[i] < SIZE || !"Tensor::getIndex: given index has out of bound values (higher then tensor SIZE)") ;
		res += p[i]*prod ;
		prod *= SIZE ;
	}
	return res ;
}


template <unsigned int SIZE, typename REAL>
void
Tensor<SIZE, REAL>::completeSymmetricTensor()
{
	std::vector<unsigned int> p ;
	p.resize(order(), 0) ;
	do
	{
		std::vector<unsigned int> sorted_p = p ;
		std::sort(sorted_p.begin(), sorted_p.begin() + (*this).order()) ;
		(*this)(p) = (*this)(sorted_p) ;
	} while (incremIndex(p)) ;
}

template <unsigned int SIZE, typename REAL>
const unsigned int&
Tensor<SIZE, REAL>::order() const
{
	return m_order ;
}

template <unsigned int SIZE, typename REAL>
unsigned int
Tensor<SIZE, REAL>::nbElem() const
{
	return pow(SIZE,m_order) ;
}

template <unsigned int SIZE, typename REAL>
REAL&
Tensor<SIZE, REAL>::operator[](unsigned int k)
{
	assert(k < nbElem() || !"Tensor::operator[] out of bound value requested") ;
	return m_data[k] ;
}

template <unsigned int SIZE, typename REAL>
const REAL&
Tensor<SIZE, REAL>::operator[](unsigned int k) const
{
	assert(k < nbElem() || !"Tensor::operator[] out of bound value requested") ;
	return m_data[k] ;
}


template <unsigned int SIZE, typename REAL>
bool
Tensor<SIZE, REAL>::incremIndex(std::vector<unsigned int>& p)
{
	int i = p.size() - 1 ;
	while (i >= 0)
	{
		p[i] = (p[i] + 1) % SIZE ;
		if (p[i] != 0) // if no overflow
			return true ;
		--i ;
	}
	return false ;
}

} /* namespace Geom */
} /* namespace CGoGN */
