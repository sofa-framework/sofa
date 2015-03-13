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

#ifndef __MR_CC_MASK__
#define __MR_CC_MASK__

#include <cmath>

namespace CGoGN
{

namespace Algo
{

namespace MR
{

namespace Primal
{

namespace Masks
{

template <typename PFP>
class CCVertexVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& m_position ;

public:
	CCVertexVertexFunctor(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		m_map.decCurrentLevel() ;

		typename PFP::VEC3 np1(0) ;
		typename PFP::VEC3 np2(0) ;
		unsigned int degree1 = 0 ;
		unsigned int degree2 = 0 ;
		Dart it = d ;
		do
		{
			++degree1 ;
			Dart dd = m_map.phi1(it) ;
			np1 += m_position[dd] ;
			Dart end = m_map.phi_1(it) ;
			dd = m_map.phi1(dd) ;
			do
			{
				++degree2 ;
				np2 += m_position[dd] ;
				dd = m_map.phi1(dd) ;
			} while(dd != end) ;
			it = m_map.phi2(m_map.phi_1(it)) ;
		} while(it != d) ;

		float beta = 3.0 / (2.0 * degree1) ;
		float gamma = 1.0 / (4.0 * degree2) ;
		np1 *= beta / degree1 ;
		np2 *= gamma / degree2 ;

		typename PFP::VEC3 vp = m_position[d] ;
		vp *= 1.0 - beta - gamma ;

		m_map.incCurrentLevel() ;

		m_position[d] = np1 + np2 + vp ;

		return false ;
	}
} ;

template <typename PFP>
class CCEdgeVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& m_position ;

public:
	CCEdgeVertexFunctor(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		Dart d1 = m_map.phi2(d) ;

		m_map.decCurrentLevel() ;

		Dart d2 = m_map.phi2(d1) ;
		Dart d3 = m_map.phi_1(d1) ;
		Dart d4 = m_map.phi_1(d2) ;
		Dart d5 = m_map.phi1(m_map.phi1(d1)) ;
		Dart d6 = m_map.phi1(m_map.phi1(d2)) ;

		typename PFP::VEC3 p1 = m_position[d1] ;
		typename PFP::VEC3 p2 = m_position[d2] ;
		typename PFP::VEC3 p3 = m_position[d3] ;
		typename PFP::VEC3 p4 = m_position[d4] ;
		typename PFP::VEC3 p5 = m_position[d5] ;
		typename PFP::VEC3 p6 = m_position[d6] ;

		p1 *= 3.0 / 8.0 ;
		p2 *= 3.0 / 8.0 ;
		p3 *= 1.0 / 16.0 ;
		p4 *= 1.0 / 16.0 ;
		p5 *= 1.0 / 16.0 ;
		p6 *= 1.0 / 16.0 ;

		m_map.incCurrentLevel() ;

		m_position[d] = p1 + p2 + p3 + p4 + p5 + p6 ;

		return false ;
	}
} ;

template <typename PFP>
class CCFaceVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& m_position ;

public:
	CCFaceVertexFunctor(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		Dart df = m_map.phi1(m_map.phi1(d)) ;

		m_map.decCurrentLevel() ;

		typename PFP::VEC3 p(0) ;
		unsigned int degree = 0 ;
		Traversor2FV<typename PFP::MAP> trav(m_map, df) ;
		for(Dart it = trav.begin(); it != trav.end(); it = trav.next())
		{
			++degree ;
			p += m_position[it] ;
		}
		p /= degree ;

		m_map.incCurrentLevel() ;

		m_position[d] = p ;

		return false ;
	}
} ;


} // namespace Masks

} // namespace Primal

} // namespace MR

} // namespace Algo

} // namespace CGoGN

#endif

