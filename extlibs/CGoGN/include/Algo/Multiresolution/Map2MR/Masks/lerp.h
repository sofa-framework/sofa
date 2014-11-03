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

#ifndef __MR_LERP_MASK__
#define __MR_LERP_MASK__

#include <cmath>
//#include "Algo/Decimation/decimation.h"

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
class LerpVertexVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	LerpVertexVertexFunctor(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		std::cout << "dartIndex(d) = " << m_map.dartIndex(d) << std::endl;
		m_map.decCurrentLevel() ;
		std::cout << "dartIndex(d) = " << m_map.dartIndex(d) << std::endl;
		typename PFP::VEC3 p = m_position[d] ;
		std::cout << "p = " << p << std::endl;
		m_map.incCurrentLevel() ;

		m_position[d] = p ;

		return false ;
	}
} ;

template <typename PFP>
class LerpEdgeVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	LerpEdgeVertexFunctor(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		Dart d1 = m_map.phi2(d) ;

		m_map.decCurrentLevel() ;
		Dart d2 = m_map.phi2(d1) ;
		typename PFP::VEC3 p = (m_position[d1] + m_position[d2]) / 2.0 ;
		m_map.incCurrentLevel() ;

		m_position[d] = p ;

		return false ;
	}
} ;

template <typename PFP>
class LerpFaceVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	LerpFaceVertexFunctor(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
        Dart df = m_map.phi1(m_map.phi1(d)) ;
        //Dart df = m_map.phi1(d) ;

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

