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

#ifndef __2MR_LERP_FILTER__
#define __2MR_LERP_FILTER__

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
 *                           SYNTHESIS FILTERS
 *********************************************************************************/

// Quad refinement
template <typename PFP>
class LerpQuadOddSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	LerpQuadOddSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorF<MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
			VEC3 vf(0.0);
			VEC3 ef(0.0);

			unsigned int count = 0;
			Traversor2FE<MAP> travFE(m_map, d);
			for (Dart dit = travFE.begin(); dit != travFE.end(); dit = travFE.next())
			{
				vf += m_position[dit];
				m_map.incCurrentLevel();
				ef += m_position[m_map.phi1(dit)];
				m_map.decCurrentLevel();
				++count;
			}            

			ef /= count;
			ef *= 2.0;

			vf /= count;

			m_map.incCurrentLevel() ;
			Dart midF = m_map.phi1(m_map.phi1(d));
            m_position[midF] += vf + ef ;
			m_map.decCurrentLevel() ;
            break;
		}

		TraversorE<MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{
			VEC3 ve = (m_position[d] + m_position[m_map.phi1(d)]) * typename PFP::REAL(0.5);

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi1(d) ;
			m_position[midV] += ve ;
			m_map.decCurrentLevel() ;
		}
	}
};

// Tri/quad refinement
template <typename PFP>
class LerpTriQuadOddSynthesisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	LerpTriQuadOddSynthesisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorF<MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
			if(m_map.faceDegree(d) != 3)
			{
				VEC3 vf(0.0);
				VEC3 ef(0.0);

				unsigned int count = 0;
				Traversor2FE<MAP> travFE(m_map, d);
				for (Dart dit = travFE.begin(); dit != travFE.end(); dit = travFE.next())
				{
					vf += m_position[dit];
					m_map.incCurrentLevel();
					ef += m_position[m_map.phi1(dit)];
					m_map.decCurrentLevel();
					++count;
				}
				ef /= count;
				ef *= 2.0;

				vf /= count;

				m_map.incCurrentLevel() ;
				Dart midF = m_map.phi1(m_map.phi1(d));
				m_position[midF] += vf + ef ;
				m_map.decCurrentLevel() ;
			}
		}

		TraversorE<MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{
			VEC3 ve = (m_position[d] + m_position[m_map.phi1(d)]) * typename PFP::REAL(0.5);

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi1(d) ;
			m_position[midV] += ve ;
			m_map.decCurrentLevel() ;
		}
	}
};

/*********************************************************************************
 *                           ANALYSIS FILTERS
 *********************************************************************************/

// Quad refinement
template <typename PFP>
class LerpQuadOddAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	LerpQuadOddAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{
			VEC3 ve = (m_position[d] + m_position[m_map.phi1(d)]) * typename PFP::REAL(0.5);

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi1(d) ;
			m_position[midV] -= ve ;
			m_map.decCurrentLevel() ;
		}

		TraversorF<MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
			VEC3 vf(0.0);
			VEC3 ef(0.0);

			unsigned int count = 0;
			Traversor2FE<MAP> travFE(m_map, d);
			for (Dart dit = travFE.begin(); dit != travFE.end(); dit = travFE.next())
			{
				vf += m_position[dit];
				m_map.incCurrentLevel();
				ef += m_position[m_map.phi1(dit)];
				m_map.decCurrentLevel();
				++count;
			}
			ef /= count;
			ef *= 2.0;

			vf /= count;

			m_map.incCurrentLevel() ;
			Dart midF = m_map.phi1(m_map.phi1(d));
            m_position[midF] -= vf + ef ;
			m_map.decCurrentLevel() ;
		}
	}
};

// Tri/quad refinement
template <typename PFP>
class LerpTriQuadOddAnalysisFilter : public Algo::MR::Filter
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;

public:
	LerpTriQuadOddAnalysisFilter(MAP& m, VertexAttribute<VEC3, MAP>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<MAP> travE(m_map) ;
		for (Dart d = travE.begin(); d != travE.end(); d = travE.next())
		{
			VEC3 ve = (m_position[d] + m_position[m_map.phi1(d)]) * typename PFP::REAL(0.5);

			m_map.incCurrentLevel() ;
			Dart midV = m_map.phi1(d) ;
			m_position[midV] -= ve ;
			m_map.decCurrentLevel() ;
		}

		TraversorF<MAP> travF(m_map) ;
		for (Dart d = travF.begin(); d != travF.end(); d = travF.next())
		{
			if(m_map.faceDegree(d) != 3)
			{
				VEC3 vf(0.0);
				VEC3 ef(0.0);

				unsigned int count = 0;
				Traversor2FE<MAP> travFE(m_map, d);
				for (Dart dit = travFE.begin(); dit != travFE.end(); dit = travFE.next())
				{
					vf += m_position[dit];
					m_map.incCurrentLevel();
					ef += m_position[m_map.phi1(dit)];
					m_map.decCurrentLevel();
					++count;
				}
				ef /= count;
				ef *= 2.0;

				vf /= count;

				m_map.incCurrentLevel() ;
				Dart midF = m_map.phi1(m_map.phi1(d));
				m_position[midF] -= vf + ef ;
				m_map.decCurrentLevel() ;
			}
		}
	}
};

} // namespace Filters

} // namespace Primal

} // namespace MR

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif

