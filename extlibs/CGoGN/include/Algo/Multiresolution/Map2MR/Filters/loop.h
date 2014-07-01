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

#ifndef __2MR_LOOP_FILTER__
#define __2MR_LOOP_FILTER__

#include <cmath>
#include "Algo/Multiresolution/filter.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MR
{

namespace Primal
{

namespace Filters
{


/*********************************************************************************
 *                           LOOP BASIC FUNCTIONS
 *********************************************************************************/

template <typename PFP>
typename PFP::VEC3 loopOddVertex(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	Dart d1)
{
	Dart d2 = map.phi2(d1) ;
	Dart d3 = map.phi_1(d1) ;
	Dart d4 = map.phi_1(d2) ;

	typename PFP::VEC3 p1 = position[d1] ;
	typename PFP::VEC3 p2 = position[d2] ;
	typename PFP::VEC3 p3 = position[d3] ;
	typename PFP::VEC3 p4 = position[d4] ;

	p1 *= 3.0 / 8.0 ;
	p2 *= 3.0 / 8.0 ;
	p3 *= 1.0 / 8.0 ;
	p4 *= 1.0 / 8.0 ;

	return p1 + p2 + p3 + p4 ;
}

template <typename PFP>
typename PFP::VEC3 loopEvenVertex(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	Dart d)
{
	map.incCurrentLevel() ;

	typename PFP::VEC3 np(0) ;
	unsigned int degree = 0 ;
	Traversor2VVaE<typename PFP::MAP> trav(map, d) ;
	for(Dart it = trav.begin(); it != trav.end(); it = trav.next())
	{
		++degree ;
		np += position[it] ;
	}

	map.decCurrentLevel() ;

	float mu = 3.0/8.0 + 1.0/4.0 * cos(2.0 * M_PI / degree) ;
	mu = (5.0/8.0 - (mu * mu)) / degree ;
	np *= 8.0/5.0 * mu ;

	return np ;
}

/*********************************************************************************
 *                           ANALYSIS FILTERS
 *********************************************************************************/

template <typename PFP>
class LoopOddAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	LoopOddAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			VEC3 p = loopOddVertex<PFP>(m_map, m_position, d) ;

			m_map.incCurrentLevel() ;

			Dart oddV = m_map.phi2(d) ;
			m_position[oddV] -= p ;

			m_map.decCurrentLevel() ;
		}
	}
} ;

template <typename PFP>
class LoopEvenAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position;

public:
	LoopEvenAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			VEC3 p = loopEvenVertex<PFP>(m_map, m_position, d) ;
			m_position[d] -= p ;
		}
	}
} ;

template <typename PFP>
class LoopNormalisationAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position;

public:
	LoopNormalisationAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			unsigned int degree = m_map.vertexDegree(d) ;
			float n = 3.0/8.0 + 1.0/4.0 * cos(2.0 * M_PI / degree) ;
			n = 8.0/5.0 * (n * n) ;

			m_position[d] /= n ;
		}
	}
} ;

/*********************************************************************************
 *                           SYNTHESIS FILTERS
 *********************************************************************************/

template <typename PFP>
class LoopOddSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position;

public:
	LoopOddSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			VEC3 p = loopOddVertex<PFP>(m_map, m_position, d) ;

			m_map.incCurrentLevel() ;

			Dart oddV = m_map.phi2(d) ;
			m_position[oddV] += p ;

			m_map.decCurrentLevel() ;
		}
	}
} ;

template <typename PFP>
class LoopEvenSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	LoopEvenSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			VEC3 p = loopEvenVertex<PFP>(m_map, m_position, d) ;
			m_position[d] += p ;
		}
	}
} ;

template <typename PFP>
class LoopNormalisationSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	LoopNormalisationSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			unsigned int degree = m_map.vertexDegree(d) ;
			float n = 3.0/8.0 + 1.0/4.0 * cos(2.0 * M_PI / degree) ;
			n = 8.0/5.0 * (n * n) ;

			m_position[d] *= n ;
		}
	}
} ;

} // namespace Filters

} // namespace Primal

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif

