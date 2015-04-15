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

#ifndef __MR_LOOP_MASK__
#define __MR_LOOP_MASK__

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
class LoopVertexVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& m_position ;

public:
	LoopVertexVertexFunctor(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		m_map.decCurrentLevel() ;

        typename PFP::VEC3 np(0, 0, 0) ;
		unsigned int degree = 0 ;
		Traversor2VVaE<typename PFP::MAP> trav(m_map, d) ;
		for(Dart it = trav.begin(); it != trav.end(); it = trav.next())
		{
			++degree ;
			np += m_position[it] ;
		}
		float tmp = 3.0 + 2.0 * cos(2.0 * M_PI / degree) ;
		float beta = (5.0 / 8.0) - ( tmp * tmp / 64.0 ) ;
		np *= beta / degree ;

		typename PFP::VEC3 vp = m_position[d] ;
		vp *= 1.0 - beta ;

		m_map.incCurrentLevel() ;

		m_position[d] = np + vp ;

		return false ;
	}
} ;

template <typename PFP>
class LoopEdgeVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& m_position ;

public:
	LoopEdgeVertexFunctor(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		Dart d1 = m_map.phi2(d) ;

		m_map.decCurrentLevel() ;

		Dart d2 = m_map.phi2(d1) ;
		Dart d3 = m_map.phi_1(d1) ;
		Dart d4 = m_map.phi_1(d2) ;

		typename PFP::VEC3 p1 = m_position[d1] ;
		typename PFP::VEC3 p2 = m_position[d2] ;
		typename PFP::VEC3 p3 = m_position[d3] ;
		typename PFP::VEC3 p4 = m_position[d4] ;

		p1 *= 3.0 / 8.0 ;
		p2 *= 3.0 / 8.0 ;
		p3 *= 1.0 / 8.0 ;
		p4 *= 1.0 / 8.0 ;

		m_map.incCurrentLevel() ;

		m_position[d] = p1 + p2 + p3 + p4 ;

		return false ;
	}
} ;

} // namespace Masks

} // namespace Primal

} // namespace MR

} // namespace Algo

} // namespace CGoGN

#endif

